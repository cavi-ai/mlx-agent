# Serving

Names for the local serving endpoint. Weights stay on the machine that runs the endpoint.

## Language

**Memory policy**:
The reviewed limits on headroom, idle residency, execution wait, and model budget for one endpoint.
_Avoid_: memory settings, headroom config

**Serving plan**:
The reviewed description of one endpoint before it is started, including the preview hash.
_Avoid_: launch config, start request

**Worker transport**:
How the endpoint reaches its model worker: a Unix listener, a loopback key, or a resident task worker.
_Avoid_: private worker, upstream

**Residency**:
Whether the endpoint's model weights are loaded, reserved, or released.
_Avoid_: lifecycle, worker state

**Identity pass**:
The check that every bound file and runtime still matches the serving plan.
_Avoid_: fingerprint check, revalidation
