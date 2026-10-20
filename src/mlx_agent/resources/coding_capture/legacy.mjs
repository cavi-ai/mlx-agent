// Supported legacy hooks are explicit; policy hooks are never silently removed.
import {$} from "bun";
import {isDeepStrictEqual} from "node:util";

const supported = new Set(["config", "tool", "event", "tool.execute.before"]);

function requireSupported(condition, message) {if (!condition) throw new Error(message)}

function checkConfiguration(before, after) {
  const {provider: beforeProviders, ...beforeRest}=before;
  const {provider: afterProviders, ...afterRest}=after;
  requireSupported(isDeepStrictEqual(beforeRest,afterRest),"Legacy configuration hook changed unsupported policy or configuration inputs");
  for (const [id, provider] of Object.entries(afterProviders ?? {})) {
    if (isDeepStrictEqual(provider,beforeProviders?.[id])) continue;
    requireSupported(provider && provider.npm === "@ai-sdk/openai-compatible",
      "Legacy configuration hook selected an unsupported provider package");
    for(const model of Object.values(provider.models ?? {}))
      requireSupported(!model.provider?.npm,"Legacy hook selected a model-specific package");
  }
}

export function legacyPlugin({id,module,objectSchema,toJSONSchema}) {
  let state;
  return {
    id,
    async bootstrap(configuration, directory) {
      requireSupported(!state,"Legacy extension bootstrap was repeated");
      const exports=Object.values(module);
      requireSupported(exports.length===1 && typeof exports[0]==="function",
        "Legacy extension must export one unambiguous factory");
      const host=new Proxy({$:$.cwd(directory)}, {get(target,key) {
        requireSupported(key in target,"Unsupported legacy factory context");return target[key];
      }});
      const hooks=await exports[0](host);
      requireSupported(hooks && typeof hooks==="object" && !Array.isArray(hooks),"Invalid legacy hook definition");
      for (const [name,hook] of Object.entries(hooks)) {
        requireSupported(supported.has(name),"Unsupported legacy hook: "+name);
        requireSupported(name==="tool" ? hook && typeof hook==="object" : typeof hook==="function","Invalid legacy hook: "+name);
      }
      const tools=Object.entries(hooks.tool ?? {}).map(([name,tool]) => {
        requireSupported(typeof tool?.execute==="function" && typeof tool.description==="string"
          && typeof objectSchema==="function" && typeof toJSONSchema==="function","Unsupported legacy tool definition");
        const schema=objectSchema(tool.args);
        requireSupported(typeof schema.parse==="function","Legacy tool needs an input validator");
        return {name,description:tool.description,input:toJSONSchema(schema),schema,execute:tool.execute};
      });
      const effective=structuredClone(configuration);
      if (hooks.config) {
        await hooks.config(effective);
        checkConfiguration(configuration,effective);
      }
      state={hooks,tools,directory};
      return effective;
    },
    async invoke(method,params={}) {
      requireSupported(state,"Owned legacy bootstrap context is missing");
      const {hooks,tools,directory}=state;
      if(method==="describe") return {id,directory,hooks:Object.keys(hooks),
        tools:tools.map(({name,description,input})=>({name,description,input}))};
      if(method==="tool") {
        const tool=tools.find(tool=>tool.name===params.name);
        requireSupported(tool,"Unknown legacy tool");
        const content=await tool.execute(tool.schema.parse(params.input),{...params.context,directory});
        requireSupported(typeof content==="string","Unsupported legacy tool result");
        return {content};
      }
      if(method==="event" && hooks.event) {await hooks.event({event:params.event});return null}
      if(method==="before" && hooks["tool.execute.before"]) {
        const event=params.event, legacyTool=event.tool==="shell" ? "bash" : event.tool;
        const input={...event,tool:legacyTool},output={args:event.input};
        await hooks["tool.execute.before"](input,output);
        return {input:output.args,tool:input.tool===legacyTool ? event.tool : input.tool};
      }
      throw new Error("Unsupported legacy operation");
    },
  };
}
