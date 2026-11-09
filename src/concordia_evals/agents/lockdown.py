"""Network lockdown solver for agent sandboxes.

Benchmarks that forbid the agent from using the network (e.g. to prevent
looking up answers) run their setup with internet access, then apply this
solver right before the agent starts. It disconnects the sample's sandbox
container(s) from every Docker network, leaving only loopback.

This works with sandbox_agent_bridge-based agents (inspect_swe CLIs): the
bridge relays model calls over the docker exec channel and the container's
loopback interface, so the agent keeps working fully offline. It is strictly
stronger than a domain allowlist: no egress exists at all.

Container discovery uses the sample's compose file path (a stable,
sample-unique handle) passed via sample metadata under the key
"compose_file" — inspect's compose-managed containers carry the
`com.docker.compose.project.config_files` label with exactly this path.
Requires the Docker SDK (host-side); skips with a warning when the metadata
or containers are not found (e.g. non-docker sandboxes).
"""

import logging
from typing import Any

from inspect_ai.solver import Generate, Solver, TaskState, solver

logger = logging.getLogger(__name__)

_LABEL_CONFIG_FILES = "com.docker.compose.project.config_files"

COMPOSE_FILE_METADATA_KEY = "compose_file"


def _find_sample_containers(compose_file: str) -> list[Any]:
    import docker

    client = docker.from_env()
    containers: list[Any] = client.containers.list(
        filters={"label": [f"{_LABEL_CONFIG_FILES}={compose_file}"]}
    )
    return containers


def disconnect_container(container: Any) -> list[str]:
    """Disconnect a container from all of its networks. Returns network names."""
    container.reload()
    networks = list(container.attrs["NetworkSettings"]["Networks"].keys())
    for network_name in networks:
        network = container.client.networks.get(network_name)
        network.disconnect(container.id, force=True)
    return networks


@solver
def network_lockdown(enabled: bool = True) -> Solver:
    """Disconnect the sample sandbox container(s) from all Docker networks.

    Args:
        enabled: If False, does nothing (convenient for task-level flags).
    """

    async def solve(state: TaskState, generate: Generate) -> TaskState:
        if not enabled:
            return state

        compose_file = state.metadata.get(COMPOSE_FILE_METADATA_KEY)
        if not isinstance(compose_file, str) or not compose_file:
            logger.warning(
                "network_lockdown: sample metadata has no %r; skipping lockdown",
                COMPOSE_FILE_METADATA_KEY,
            )
            return state

        containers = _find_sample_containers(compose_file)
        if not containers:
            logger.warning("network_lockdown: no containers found for %s; skipping", compose_file)
            return state

        for container in containers:
            networks = disconnect_container(container)
            logger.info("network_lockdown: disconnected %s from %s", container.name, networks)
        return state

    return solve
