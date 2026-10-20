"""Explicit installed-client static compiler proof; never installs a runtime."""

import hashlib
import importlib
import json
import os
from pathlib import Path
import subprocess
import tempfile
import unittest

from mlx_agent.task_contracts import TaskError

try:
    plugins = importlib.import_module("mlx_agent.coding_capture_plugins")
except ModuleNotFoundError:
    plugins = None

CLIENT = os.environ.get("MLX_AGENT_OPENCODE_TEST_CLIENT")


@unittest.skipUnless(CLIENT, "Set an existing OpenCode executable for the opt-in compiler proof.")
class InstalledCompilerTests(unittest.TestCase):
    def setUp(self):
        self.assertIsNotNone(plugins, "Extension compiler is missing")
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name).resolve()
        self.source = self.root / "source"
        self.source.mkdir()

    def compile(self, text):
        entry = self.source / "plugin.ts"
        entry.write_text(text)
        return plugins.freeze_extensions(Path(CLIENT), [str(entry)], self.root / "frozen")

    def test_compilation_never_evaluates_imported_code_and_binds_package_files(self):
        marker = self.root / "evaluated"
        package = self.source / "node_modules" / "existing"
        package.mkdir(parents=True)
        (package / "package.json").write_text(json.dumps({"name": "existing", "type": "module", "exports": "./index.js"}))
        (package / "index.js").write_text("export {value} from './value.js'")
        (package / "value.js").write_text("export const value = 17")
        result = self.compile("import {writeFileSync} from 'node:fs'; import {value} from 'existing';"
                              f"writeFileSync({json.dumps(str(marker))},'bad');"
                              "export default {id:'fixture',setup(){return value}}")
        self.assertFalse(marker.exists())
        names = {Path(item["path"]).name for item in result["inputs"]}
        self.assertTrue({"plugin.ts", "package.json", "index.js", "value.js"} <= names)
        bundle = Path(result["bundles"][0]["path"])
        self.assertEqual(result["bundles"][0]["sha256"], hashlib.sha256(bundle.read_bytes()).hexdigest())
        self.assertNotIn("from \"existing\"", bundle.read_text())
        plugins.verify_extensions(result)
        (package / "value.js").write_text("export const value = 99")
        with self.assertRaises(TaskError) as failure:
            plugins.verify_extensions(result)
        self.assertEqual(failure.exception.code, "capture_context_changed")

    def test_missing_packages_dynamic_imports_and_macros_fail_without_execution(self):
        marker = self.root / "macro-ran"
        macro = self.source / "macro.ts"
        macro.write_text("import {writeFileSync} from 'node:fs'; export function bad(){"
                         f"writeFileSync({json.dumps(str(marker))},'bad');return 'bad'" + "}")
        fixtures = ["import 'not-installed-capture-fixture'; export default {}",
                    "const name=process.env.PACKAGE; export const load=()=>import(name)",
                    "import {bad} from './macro.ts' with {type:'macro'}; export const value=bad()"]
        for index, text in enumerate(fixtures):
            with self.subTest(index=index):
                entry = self.source / f"failure-{index}.ts"
                entry.write_text(text)
                with self.assertRaises(TaskError):
                    plugins.freeze_extensions(Path(CLIENT), [str(entry)], self.root / f"rejected-{index}")
                self.assertFalse(marker.exists())

    def test_original_import_meta_url_survives_owned_bundle(self):
        result = self.compile("export default {id:'fixture',setup(){},source:import.meta.url}")
        bundle = result["bundles"][0]["path"]
        environment = dict(os.environ, BUN_BE_BUN="1")
        run = subprocess.run([CLIENT, "--no-install", "--no-env-file", "-e",
                              "const p=await import(process.argv[1]);console.log(p.default.source)", bundle],
                             env=environment, capture_output=True, text=True, timeout=15)
        self.assertEqual(run.returncode, 0, run.stderr)
        self.assertEqual(run.stdout.strip(), (self.source / "plugin.ts").as_uri())

    def test_imported_json_is_bound_and_unsupported_loaders_are_refused(self):
        data = self.source / "settings.json"
        data.write_text('{"value":17}')
        result = self.compile("import settings from './settings.json'; export default {id:'fixture',setup(){return settings.value}}")
        self.assertIn(str(data), [item["path"] for item in result["inputs"]])
        data.write_text('{"value":99}')
        with self.assertRaises(TaskError):
            plugins.verify_extensions(result)
        asset = self.source / "unbound.txt"
        asset.write_text("asset")
        entry = self.source / "unsupported.ts"
        entry.write_text("import value from './unbound.txt';export default {id:'fixture',setup(){return value}}")
        with self.assertRaises(TaskError):
            plugins.freeze_extensions(Path(CLIENT), [str(entry)], self.root / "unsupported-output")

    def test_changed_client_symlink_requires_new_preview(self):
        selected=self.root/"selected-client"
        selected.symlink_to(Path(CLIENT).resolve())
        entry=self.source/"plugin.ts"
        entry.write_text("export default {id:'fixture',setup(){}}")
        result=plugins.freeze_extensions(selected,[str(entry)],self.root/"frozen")
        replacement=self.root/"different-client"
        replacement.write_text("not an executable")
        selected.unlink()
        selected.symlink_to(replacement)
        with self.assertRaises(TaskError) as failure:
            plugins.verify_extensions(result)
        self.assertEqual(failure.exception.code,"capture_context_changed")

    def test_legacy_tools_provider_updates_hooks_and_cleanup_use_v2_contract(self):
        project = Path(__file__).resolve().parents[2]
        bridge = project / "src/mlx_agent/resources/coding_capture/legacy.mjs"
        self.assertTrue(bridge.is_file(), "Legacy hook bridge is missing")
        fixture = project / "tests/fixtures/coding_capture_legacy_contract.mjs"
        for mode in ("tool", "config", "events", "reject"):
            with self.subTest(mode=mode):
                run = subprocess.run([CLIENT, "--no-install", "--no-env-file", str(fixture), str(bridge), mode],
                                     env=dict(os.environ, BUN_BE_BUN="1"), cwd=self.root,
                                     capture_output=True, text=True, timeout=15)
                self.assertEqual(run.returncode, 0, run.stderr)
                self.assertEqual(json.loads(run.stdout)["status"], "passed")

    def test_named_legacy_factory_is_bootstrapped_once_and_exposes_v2_definition(self):
        result = self.compile("let calls=0; export const Legacy=async()=>{calls++;return {config:async c=>{"
                              "c.provider={local:{npm:'@ai-sdk/openai-compatible',name:'From hook',models:{x:{name:'X'}}}}"
                              "}}}")
        bundle = result["bundles"][0]
        self.assertTrue(bundle["legacy"])
        code = ("const p=(await import(process.argv[1])).default;"
                "const c=await p.bootstrap({permission:{shell:'ask'}},process.cwd());"
                "await p.invoke('describe');"
                "console.log(JSON.stringify({id:p.id,config:c}));")
        run = subprocess.run([CLIENT,"--no-install","--no-env-file","-e",code,bundle["path"]],
                             env=dict(os.environ,BUN_BE_BUN="1"),cwd=self.root,capture_output=True,text=True,timeout=15)
        self.assertEqual(run.returncode,0,run.stderr)
        value=json.loads(run.stdout)
        self.assertTrue(value["id"].startswith("mlx-agent.capture."))
        self.assertEqual(value["config"]["provider"]["local"]["name"],"From hook")
        self.assertEqual(value["config"]["permission"],{"shell":"ask"})

    def test_owned_host_bootstraps_hooks_once_without_compiled_preload(self):
        result=self.compile("export const Legacy=async()=>({config:async c=>{"
                            "c.provider={local:{npm:'@ai-sdk/openai-compatible',name:'Discovered'}}}})")
        profile=self.root/"capture profile"
        with plugins.ExtensionHost(Path(CLIENT),result,profile,{"permission":{"bash":"ask"}},self.source) as owned:
            prepared=owned.profile
            effective=json.loads(Path(prepared["effective_config_path"]).read_text())
            self.assertEqual(effective["permission"],{"bash":"ask"})
            self.assertEqual(effective["provider"]["local"]["name"],"Discovered")
            receipt=json.loads(Path(prepared["receipt_path"]).read_text())
            self.assertEqual(len(receipt["plugins"]),1)
            self.assertEqual(receipt["effective_config_sha256"],hashlib.sha256(Path(prepared["effective_config_path"]).read_bytes()).hexdigest())
            self.assertEqual(prepared["environment"]["HOME"],os.environ["HOME"])
            self.assertEqual(prepared["environment"]["PWD"],str(self.source))
            self.assertTrue(Path(prepared["environment"]["OPENCODE_DB"]).is_relative_to(profile))

    def test_proxy_uses_inherited_channel_and_tool_children_do_not_inherit_it(self):
        result=self.compile("let count=0;export const Legacy=async()=>{if(++count!==1)throw Error('repeated factory');"
                            "return {'tool.execute.before':async (i,o)=>{o.args.command+=' #'+count}}}")
        profile=self.root/"private profile"
        with plugins.ExtensionHost(Path(CLIENT),result,profile,{},self.source) as owned:
            entry=self.root/"proxy-fixture.mjs"
            entry.write_text("import {proxyPlugin} from "+json.dumps(str(profile/"proxy.mjs"))+";"
                             "import {fstatSync} from 'node:fs';"
                             "const fd=Number(process.env.MLX_AGENT_CAPTURE_EXTENSION_FD),before=fstatSync(fd);let hook;"
                             "const p=proxyPlugin("+json.dumps(result["bundles"][0]["id"])+");"
                             "const clean=await p.setup({location:{directory:process.cwd()},tool:{hook:async(n,f)=>{hook=f;return {dispose:async()=>{}}}}});"
                             "const event={tool:'shell',input:{command:'git commit'}};await hook(event);"
                             "const child=Bun.spawn({cmd:[process.execPath,'--no-install','-e',"
                             "\"import{fstatSync}from'node:fs';try{const s=fstatSync(Number(process.env.PROBE_FD));console.log(JSON.stringify({socket:s.isSocket(),dev:s.dev,ino:s.ino}))}catch{console.log('{}')}\"],"
                             "env:{...process.env,BUN_BE_BUN:'1',PROBE_FD:String(fd)},stdout:'pipe',stderr:'pipe'});"
                             "const seen=await new Response(child.stdout).json();await child.exited;"
                             "const leaked=Boolean(seen.socket&&seen.dev===before.dev&&seen.ino===before.ino);"
                             "await clean();console.log(JSON.stringify({command:event.input.command,leaked}));process.exit(0)")
            run=subprocess.run([CLIENT,"--no-install","--no-env-file",str(entry)],
                               env=dict(owned.profile["environment"],BUN_BE_BUN="1"),cwd=self.source,
                               pass_fds=owned.server_fds,capture_output=True,text=True,timeout=15)
            self.assertEqual(run.returncode,0,run.stderr)
            value=json.loads(run.stdout)
            self.assertEqual(value["command"],"git commit #1")
            self.assertFalse(value["leaked"])
