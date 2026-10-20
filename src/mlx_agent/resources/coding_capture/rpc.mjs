// Anonymous inherited sockets are the only transport; there is no listener.
import {createReadStream,createWriteStream,closeSync} from "node:fs";
import {randomUUID} from "node:crypto";
const maximum=1024*1024;

export function connection(fd,handler) {
  const reader=createReadStream(null,{fd,autoClose:false}), writer=createWriteStream(null,{fd,autoClose:false}), pending=new Map();
  const prefix=randomUUID();let sequence=0, buffer=Buffer.alloc(0), inFlight=0, closed=false;
  function fail(error) {
    for(const item of pending.values()) item.reject(error);pending.clear();
    if(!closed) {closed=true;reader.destroy();writer.destroy();try {closeSync(fd)} catch {}}
  }
  function send(value) {
    const raw=Buffer.from(JSON.stringify(value)+"\n");
    if(raw.length>maximum) throw new Error("Extension message budget exceeded");
    writer.write(raw);
  }
  reader.on("error",fail);writer.on("error",fail);
  reader.on("end",()=>fail(new Error("Owned extension host disconnected")));
  reader.on("data",chunk=>{
    buffer=Buffer.concat([buffer,chunk]);
    try {
      for(let end; (end=buffer.indexOf(10))>=0;) {
        if(end>=maximum) throw new Error("Extension message budget exceeded");
        const message=JSON.parse(new TextDecoder("utf-8",{fatal:true}).decode(buffer.subarray(0,end)));
        buffer=buffer.subarray(end+1);
        if(typeof message.id!=="string" || message.id.length>64) throw new Error("Invalid extension message identity");
        if(message.method!==undefined) {
          if(!handler || ++inFlight>16) throw new Error("Unexpected extension request");
          Promise.resolve().then(()=>handler(message.method,message.params)).then(
            result=>send({id:message.id,result}),()=>send({id:message.id,error:"Extension hook failed"})
          ).catch(fail).finally(()=>{inFlight--});
        } else {
          const request=pending.get(message.id);
          if(!request) throw new Error("Unmatched extension reply");
          pending.delete(message.id);
          message.error ? request.reject(new Error(message.error)) : request.resolve(message.result);
        }
      }
      if(buffer.length>=maximum) throw new Error("Extension message budget exceeded");
    } catch(error) {fail(error)}
  });
  return {
    call(method,params) {
      return new Promise((resolve,reject)=>{
        if(closed || pending.size>=16) return reject(new Error("Owned extension host is unavailable"));
        const id=prefix+":"+(++sequence);
        pending.set(id,{resolve,reject});
        try {send({id,method,params})} catch(error) {pending.delete(id);reject(error)}
      });
    },
    close:()=>fail(new Error("Owned extension channel closed")),
  };
}
