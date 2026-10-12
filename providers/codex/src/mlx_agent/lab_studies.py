"""Reviewer-scoped native operations over the existing adaptive study service."""
import hashlib
from video_lab.adaptive_study_models import AdaptiveStudyDefinition, adaptive_study_input_fingerprint
from video_lab.adaptive_study_repository import AdaptiveStudyRepository
from video_lab.adaptive_study_service import AdaptiveStudyService
from video_lab.domain import Manifest, DisplayPreference, utc_now
from video_lab.preference_optimizer import OptimizerConfig
from mlx_agent.compute import ComputeLease

def execute(service, reviews, registry, reviewer, request, worker_factory):
    repository = AdaptiveStudyRepository(service.experiments.database)
    def bound(definition):
        return AdaptiveStudyService(repository, service, reviews,
            registry.capabilities(definition.base_manifest.adapter_id), reviewer_id=reviewer)
    operation = request["operation"]
    if operation == "study-list":
        items = []
        for definition in repository.list():
            if repository.owner(definition.id) == reviewer:
                items.append(bound(definition).show(definition.id).model_dump(mode="json"))
        return {"studies": items}
    if operation == "study-create":
        manifest = Manifest.model_validate(request["manifest"])
        config = OptimizerConfig()
        fingerprint = adaptive_study_input_fingerprint(manifest, config)
        digest = hashlib.sha256(b"video-lab.adaptive-cli-study-id.v1\0" + reviewer.encode() + b"\0" + fingerprint.encode()).hexdigest()
        definition = AdaptiveStudyDefinition(id="study_"+digest[:24], prompt=manifest.prompt,
            negative_prompt=manifest.negative_prompt, seed=manifest.seed, base_manifest=manifest,
            optimizer_config=config, input_fingerprint=fingerprint, created_at=utc_now())
        return {"study": bound(definition).create_study(definition).model_dump(mode="json")}
    definition = repository.get(request["study_id"])
    study = bound(definition)
    view = study.show(definition.id)  # Verifies reviewer binding before any action.
    if operation == "study-show":
        return {"study": view.model_dump(mode="json")}
    if operation == "study-resume":
        return {"study": study.resume(definition.id).model_dump(mode="json")}
    if operation == "study-submit":
        return {"study": study.record_preference(definition.id, request["session_id"],
                    DisplayPreference(request["preference"])).model_dump(mode="json")}
    if operation == "study-run":
        if view.next_action != "complete_generation":
            raise ValueError("Study has no generation pair awaiting execution.")
        events = repository.list_events(definition.id)
        experiment_id = events[-1].payload["experiment_id"]
        with ComputeLease("adaptive-study-generation"):
            worker = worker_factory()
            for job in service.jobs.list_for_experiment(experiment_id):
                if job.state.value == "queued":
                    worker.run_once(job.id)
        return {"study": study.resume(definition.id).model_dump(mode="json")}
    raise ValueError("Unknown adaptive operation.")
