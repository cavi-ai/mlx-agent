import assert from "node:assert/strict";
import {pathToFileURL} from "node:url";
const {legacyPlugin} = await import(pathToFileURL(process.argv[2]).href);
const {proxyPlugin}=await import(new URL("./proxy.mjs",pathToFileURL(process.argv[2])).href);
const mode = process.argv[3];
const setup=(plugin,ctx)=>proxyPlugin(plugin.id,{call:(_method,p)=>plugin.invoke(p.method,p.params)}).setup(ctx);

function host(events = []) {
  const tools = new Map(), callbacks = new Map(), providers = new Map();
  let disposed = 0;
  const register = () => ({dispose: async () => {disposed++}});
  const ctx = {
    location: {directory: process.cwd()},
    tool: {
      transform: async callback => {callback({add: tool => tools.set(tool.name, tool)});return register()},
      hook: async (name, callback) => {callbacks.set(name, callback);return register()},
    },
    provider: {transform: async callback => {
      callback({get: id => providers.get(id),
        add: ({info, models}) => providers.set(info.id, {provider:info, models:new Map(models.map(m => [m.id,m]))}),
        update: (id, update) => update(providers.get(id).provider),
        models: {set: (id, models) => {providers.get(id).models = new Map(models.map(m => [m.id,m]))}}});
      return register();
    }},
    model: {transform: async callback => {
      callback({list: id => [...providers.values()].flatMap(p => [...p.models.values()]).filter(m => !id || m.providerID===id),
        update: (pid,id,update) => update(providers.get(pid).models.get(id))});return register();
    }},
    event: {subscribe: async function* () {for (const event of events) yield event}},
  };
  return {ctx, tools, callbacks, providers, disposed: () => disposed};
}

if (mode === "tool") {
  const h = host();
  let seen;
  const plugin = legacyPlugin({id:"fixture.tool", module:{Legacy:async () => ({tool:{bounded:{
    description:"Keep the original contract", args:{value:{type:"string",maxLength:4}},
    execute:async (args,context) => {seen=[args,context.sessionID];return JSON.stringify(args)},
  }}})}, objectSchema: args => ({parse:value=>value,json:{type:"object",properties:args,required:Object.keys(args)}}),
    toJSONSchema:schema=>schema.json,configuration:{}});
  assert.equal(plugin.id,"fixture.tool");
  await plugin.bootstrap({},process.cwd());
  const cleanup = await setup(plugin,h.ctx);
  const tool = h.tools.get("bounded");
  assert.equal(tool.input.properties.value.maxLength,4);
  assert.equal(tool.description,"Keep the original contract");
  assert.equal(tool.options.codemode,false);
  assert.deepEqual(await tool.execute({value:"abc"},{sessionID:"ses_fixture"}),{content:'{"value":"abc"}'});
  assert.deepEqual(seen,[{value:"abc"},"ses_fixture"]);
  await cleanup();
  assert.equal(h.disposed(),1);
} else if (mode === "config") {
  const h=host();
  h.providers.set("local",{provider:{id:"local",name:"Original",activation:"auto",package:"native",settings:{timeout:99},headers:{original:"yes"}},models:new Map()});
  const original={permission:{bash:"ask"},provider:{local:{npm:"@ai-sdk/openai-compatible",options:{timeout:99},models:{old:{name:"Old"}}}}};
  const plugin=legacyPlugin({id:"fixture.config",configuration:original,module:{Legacy:async () => ({config:async config => {
    config.provider.local.name="Discovered";
    config.provider.local.options.baseURL="http://127.0.0.1:1234/v1";
    config.provider.local.models={new:{name:"New"}};
    config.provider.local.blacklist=["old"];
  }})}});
  const prepared=await plugin.bootstrap(original,process.cwd());
  const cleanup=await setup(plugin,h.ctx);
  assert.equal(prepared.provider.local.name,"Discovered");
  assert.equal(prepared.provider.local.npm,"@ai-sdk/openai-compatible");
  assert.deepEqual(prepared.provider.local.options,{timeout:99,baseURL:"http://127.0.0.1:1234/v1"});
  assert.deepEqual(Object.keys(prepared.provider.local.models),["new"]);
  assert.equal(prepared.provider.local.models.new.name,"New");
  assert.equal(original.provider.local.options.baseURL,undefined);
  assert.deepEqual(original.permission,{bash:"ask"});
  await cleanup();
} else if (mode === "events") {
  const seen=[], h=host([{type:"session.created",data:{id:"ses_fixture"}},
    {type:"filesystem.changed",data:{file:"graph.db",event:"change"}}]);
  const plugin=legacyPlugin({id:"fixture.events",configuration:{},module:{Legacy:async ({$}) => {
    assert.equal(typeof $, "function");
    return {event:async ({event}) => seen.push(event),
      "tool.execute.before":async (input,output) => {if(input.tool==="bash") output.args.command+=" --checked"}};
  }}});
  await plugin.bootstrap({},process.cwd());
  const cleanup=await setup(plugin,h.ctx);
  const shell={tool:"shell",input:{command:"git commit"},sessionID:"ses_fixture"};
  await h.callbacks.get("execute.before")(shell);
  assert.equal(shell.input.command,"git commit --checked");
  await h.callbacks.get("execute.after")({tool:"edit",status:"error",input:{}});
  await h.callbacks.get("execute.after")({tool:"edit",status:"completed",input:{}});
  await new Promise(resolve=>setTimeout(resolve,0));
  await cleanup();
  assert.equal(seen.filter(e=>e.type==="file.edited").length,1);
  assert.equal(seen.find(e=>e.type==="session.created").properties.id,"ses_fixture");
} else if (mode === "reject") {
  for (const hooks of [{"permission.ask":async()=>{}},
    {config:async config=>{config.permission={"*":"allow"}}},
    {config:async config=>{config.provider={new:{npm:"missing-package"}}}}]) {
    const h=host();
    const plugin=legacyPlugin({id:"fixture.reject",configuration:{},module:{Legacy:async()=>hooks}});
    await assert.rejects(plugin.bootstrap({},process.cwd()));
    assert.equal(h.tools.size,0);
    assert.equal(h.callbacks.size,0);
    assert.equal(h.providers.size,0);
  }
  const h=host();
  const absent=legacyPlugin({id:"fixture.absent",module:{Legacy:async()=>({})}});
  await assert.rejects(setup(absent,h.ctx), /bootstrap/);
} else throw new Error("Unknown contract mode");
console.log(JSON.stringify({mode,status:"passed"}));
