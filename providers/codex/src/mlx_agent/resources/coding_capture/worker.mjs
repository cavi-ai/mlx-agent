import {readFileSync,lstatSync} from "node:fs";
import {pathToFileURL} from "node:url";
import {createHash} from "node:crypto";
import {connection} from "./rpc.mjs";
const hash=raw=>createHash("sha256").update(raw).digest("hex");
const plan=JSON.parse(readFileSync(process.env.MLX_AGENT_CAPTURE_BOOTSTRAP,"utf8"));
if(plan.schema!=="coding-capture-bootstrap-plan/1") throw new Error("Invalid owned bootstrap plan");
const instances=new Map();let booted=false;
const descriptor=Number(process.env.MLX_AGENT_CAPTURE_EXTENSION_FD);
delete process.env.MLX_AGENT_CAPTURE_EXTENSION_FD;
delete process.env.MLX_AGENT_CAPTURE_BOOTSTRAP;
delete process.env.BUN_BE_BUN;
connection(descriptor,async(method,params)=>{
  if(method==="bootstrap" && !booted) {
    booted=true;
    let configuration=JSON.parse(readFileSync(plan.original,"utf8"));
    const plugins=[];
    for(const bundle of plan.bundles.filter(item=>item.legacy)) {
      if(!lstatSync(bundle.path).isFile() || hash(readFileSync(bundle.path))!==bundle.sha256)
        throw new Error("Owned extension bundle changed");
      const plugin=(await import(pathToFileURL(bundle.path).href)).default;
      if(plugin.id!==bundle.id || typeof plugin.bootstrap!=="function") throw new Error("Extension identity changed");
      configuration=await plugin.bootstrap(configuration,plan.project);
      const description=await plugin.invoke("describe");
      instances.set(plugin.id,plugin);
      plugins.push({id:plugin.id,source_sha256:bundle.sha256,description});
    }
    return {configuration,plugins};
  }
  if(method!=="extension" || !instances.has(params?.id)) throw new Error("Unknown extension operation");
  return instances.get(params.id).invoke(params.method,params.params);
});
// Keep the PID available to its owner even if the server disconnects while a
// hook has children. The owner terminates the checked group before stdin closes.
process.stdin.resume();
process.stdin.on("end",()=>process.exit(0));
