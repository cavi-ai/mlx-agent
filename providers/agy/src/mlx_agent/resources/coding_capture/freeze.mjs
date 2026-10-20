// This entrypoint runs with BUN_BE_BUN, --no-install and --no-macros.
// Compilation observes source bytes; it never imports an extension for execution.
import {readFileSync, lstatSync, realpathSync, existsSync, writeFileSync, chmodSync} from "node:fs";
import path from "node:path";
import {pathToFileURL, fileURLToPath} from "node:url";
import {createHash} from "node:crypto";

const hash = bytes => createHash("sha256").update(bytes).digest("hex");
const inputs = new Map();
let total = 0;
const request = JSON.parse(await Bun.stdin.text());
const bridge = fileURLToPath(new URL("./legacy.mjs", import.meta.url));

function bind(file) {
  const name = realpathSync(file);
  const stat = lstatSync(name, {bigint: true});
  if (!stat.isFile()) throw new Error("Nonregular extension dependency");
  const raw = readFileSync(name);
  const item = {path: name, identity: [stat.dev, stat.ino, stat.size, stat.mtimeNs].map(String), sha256: hash(raw)};
  const previous = inputs.get(name);
  if (previous && JSON.stringify(previous) !== JSON.stringify(item)) throw new Error("Dependency changed during compilation");
  if (!previous) {
    total += raw.length;
    if (inputs.size >= 4096 || total > request.maxBytes) throw new Error("Extension dependency budget exceeded");
    inputs.set(name, item);
  }
  return raw;
}

function manifests(directory) {
  for (let current = directory; ; current = path.dirname(current)) {
    const manifest = path.join(current, "package.json");
    if (existsSync(manifest)) bind(manifest);
    if (current === path.dirname(current)) break;
  }
}

function sourceText(file) {
  const raw=bind(file).toString("utf8");
  if (/\bimport(?:\s|\/\*[\s\S]*?\*\/|\/\/[^\n]*\n)*\(/.test(raw)
      || /\brequire\b/.test(raw) || /\btype\s*:\s*["']macro["']/.test(raw))
    throw new Error("Dynamic imports, require and macros need an unsupported dependency boundary");
  const loader=file.endsWith("tsx") ? "tsx" : file.endsWith("jsx") ? "jsx" : /\.[cm]?ts$/.test(file) ? "ts" : "js";
  return new Bun.Transpiler({loader,target:"bun",
    define:{"import.meta.url":JSON.stringify(pathToFileURL(file).href)}}).transformSync(raw);
}

try {
  const bundles = [];
  let bundleBytes=0;
  for (const [index, source] of request.sources.entries()) {
    const scanned=new Bun.Transpiler({loader:"js",target:"bun"}).scan(sourceText(source));
    const legacy=!scanned.exports.includes("default");
    const id="mlx-agent.capture."+hash(source+":"+inputs.get(realpathSync(source)).sha256).slice(0,24);
    let sdk;
    if (legacy && scanned.imports.some(item=>item.path==="@opencode-ai/plugin"))
      sdk=Bun.resolveSync("@opencode-ai/plugin",path.dirname(source));
    const entry=legacy ? [
      `import * as module from ${JSON.stringify(source)};`,
      `import {legacyPlugin} from ${JSON.stringify(bridge)};`,
      sdk ? `import {tool} from ${JSON.stringify(sdk)};` : "",
      `export default legacyPlugin({id:${JSON.stringify(id)},module,objectSchema:${sdk ? "tool.schema.object" : "undefined"},toJSONSchema:${sdk ? "tool.schema.toJSONSchema" : "undefined"}});`,
    ].join("\n") : `export {default} from ${JSON.stringify(source)};`;
    const result = await Bun.build({
      entrypoints: ["capture-entry"], target: "bun", format: "esm", sourcemap: "none", write: false,
      plugins: [{name: "capture-local-only", setup(build) {
        build.onResolve({filter: /.*/}, args => {
          if (args.path==="capture-entry") return {path:args.path,namespace:"capture"};
          if (args.path.startsWith("node:") || args.path === "bun") return {path: args.path, external: true};
          const resolved = Bun.resolveSync(args.path, args.resolveDir || path.dirname(source));
          manifests(path.dirname(resolved));
          return {path: resolved};
        });
        build.onLoad({filter:/.*/,namespace:"capture"},()=>({contents:entry,loader:"js",resolveDir:path.dirname(source)}));
        build.onLoad({filter: /.*/}, args => {
          if (/\.[cm]?[jt]sx?$/.test(args.path)) return {contents:sourceText(args.path),loader:"js"};
          if (args.path.endsWith(".json")) return {contents:bind(args.path).toString("utf8"),loader:"json"};
          throw new Error("Unsupported local extension dependency format");
        });
      }}],
    });
    if (!result.success || result.outputs.length !== 1) throw new Error("Unresolved static imports");
    const bytes = new Uint8Array(await result.outputs[0].arrayBuffer());
    bundleBytes+=bytes.length;
    if (bundleBytes > request.maxBytes) throw new Error("Compiled extension budget exceeded");
    const destination = path.join(request.output, `extension-${index}.mjs`);
    writeFileSync(destination, bytes, {flag: "wx", mode: 0o600});
    chmodSync(destination, 0o600);
    const stat = lstatSync(destination, {bigint: true});
    bundles.push({path: destination, identity: [stat.dev, stat.ino, stat.size, stat.mtimeNs].map(String),
      sha256: hash(bytes), bytes: bytes.length, source, legacy, id});
  }
  const reply = JSON.stringify({schema: "coding-capture-extensions/1", inputs: [...inputs.values()], bundles});
  if (Buffer.byteLength(reply) > 1024 * 1024) throw new Error("Extension manifest budget exceeded");
  process.stdout.write(reply);
} catch (error) {
  process.stderr.write(String(error?.message || "Cannot freeze local extensions").slice(0, 4096));
  process.exitCode = 1;
}
