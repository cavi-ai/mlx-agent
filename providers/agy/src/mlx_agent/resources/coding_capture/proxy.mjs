import {connection} from "./rpc.mjs";
const key=Symbol.for("mlx-agent.coding-capture.rpc/1");
function ownedConnection() {
  if(!globalThis[key]) {
    const fd=Number(process.env.MLX_AGENT_CAPTURE_EXTENSION_FD);
    if(!Number.isInteger(fd) || fd<3) throw new Error("Owned extension channel is missing");
    globalThis[key]=connection(fd);
    delete process.env.MLX_AGENT_CAPTURE_EXTENSION_FD;
  }
  return globalThis[key];
}

export function proxyPlugin(id,transport) {
  return {id,async setup(ctx) {
    const rpc=transport ?? ownedConnection();
    const call=(method,params={})=>rpc.call("extension",{id,method,params});
    const description=await call("describe");
    if(description.id!==id || description.directory!==ctx.location.directory) throw new Error("Extension project changed");
    const registrations=[], controller=new AbortController();let pump=Promise.resolve();
    try {
      if(description.tools.length) registrations.push(await ctx.tool.transform(editor=>{
        for(const tool of description.tools) editor.add({...tool,options:{codemode:false},execute:(input,context)=>call("tool",{name:tool.name,input,
          context:{sessionID:context.sessionID,agent:context.agent,messageID:context.messageID,id:context.id}})});
      }));
      if(description.hooks.includes("tool.execute.before")) registrations.push(await ctx.tool.hook("execute.before",async event=>{
        const result=await call("before",{event});event.input=result.input;event.tool=result.tool;
      }));
      if(description.hooks.includes("event")) {
        // Using filesystem writes here would feed graph updates back into themselves.
        registrations.push(await ctx.tool.hook("execute.after",async event=>{
          if(event.status==="completed" && ["edit","write","patch"].includes(event.tool))
            await call("event",{event:{type:"file.edited",properties:{}}});
        }));
        pump=(async()=>{
          for await(const event of ctx.event.subscribe({signal:controller.signal})) {
            if(controller.signal.aborted) break;
            await call("event",{event:{type:event.type,properties:event.data ?? {}}});
          }
        })();
        void pump.catch(()=>{});
      }
    } catch(error) {
      controller.abort();await Promise.allSettled(registrations.map(item=>item.dispose()));throw error;
    }
    return async()=>{controller.abort();await Promise.all(registrations.map(item=>item.dispose()));await pump};
  }};
}
