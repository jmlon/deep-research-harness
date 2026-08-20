"""Load `config.yaml` (PRD §13) into typed settings for the harness.

Path resolution model (stage 2 — pipx-installable tool): the tool is installed once, globally,
and then run from within a researcher's own **project folder** — a plain directory with no
special structure required. Config discovery and all relative paths (notably `output.report_dir`)
resolve against that project folder (the current working directory), never against wherever the
package itself is installed.

Config lookup order (first match wins):
1. An explicit `--config <path>` passed on the command line.
2. `./config.yaml` in the current working directory (the project folder).
3. The bundled default shipped inside the package (`deep_research/default_config.yaml`) — lets
   the tool run with sensible defaults before a project has been `init`-ed.

Model strings are then overridable from the environment (see `MODEL_ENV_VAR` below), so the
harness runs against whichever provider you have a key for without editing any file. Nothing
here is provider-specific: a model string is `"<provider>:<model-name>"` and Pydantic AI
resolves the provider and its API key from it.
"""

from __future__ import annotations

import os
from decimal import Decimal
from importlib import resources
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator

PROJECT_CONFIG_NAME = "config.yaml"

MODEL_ENV_VAR = "DEEP_RESEARCH_MODEL"
"""Sets every agent's model at once, overriding whatever `config.yaml` says."""

ROLE_MODEL_ENV_VARS = {
    "lead": "DEEP_RESEARCH_MODEL_LEAD",
    "researcher": "DEEP_RESEARCH_MODEL_RESEARCHER",
    "critic": "DEEP_RESEARCH_MODEL_CRITIC",
}
"""Per-role overrides, which win over `MODEL_ENV_VAR` — e.g. a cheaper critic."""


class StrictModel(BaseModel):
    """Base for every config section: an unrecognized key is an error, not a shrug.

    `config.yaml` is hand-edited, and every value in it has a default. Under Pydantic's default
    `extra='ignore'`, a typo (`breadth_budgets`, `worker_concurency`) silently leaves the default
    in place — the operator sees a run that ignores the budget they set, with nothing to explain
    why. Forbidding extras turns that into an immediate, named error at startup.
    """

    model_config = ConfigDict(extra="forbid")


class ModelConfig(StrictModel):
    lead: str
    researcher: str
    critic: str


class ModelPrice(StrictModel):
    """USD per million tokens for one model, used only when `genai-prices` has no data.

    Rates, not totals: the same two numbers every provider publishes, so a price can be copied
    off a pricing page without arithmetic.
    """

    input_usd_per_1m: float = Field(ge=0, description="USD per 1M input (prompt) tokens")
    output_usd_per_1m: float = Field(ge=0, description="USD per 1M output (completion) tokens")

    def cost_usd(self, input_tokens: int, output_tokens: int) -> Decimal:
        """Cost of one call, in USD.

        `Decimal` throughout: these accumulate across a run and are compared against a spend
        ceiling, and float drift in money is not worth the risk.
        """
        per_million = Decimal(1_000_000)
        return (
            Decimal(str(self.input_usd_per_1m)) * Decimal(input_tokens) / per_million
            + Decimal(str(self.output_usd_per_1m)) * Decimal(output_tokens) / per_million
        )


class SearchConfig(StrictModel):
    backend: str = "duckduckgo"


def _is_set(model: BaseModel, key: str) -> bool:
    """Whether `key` carries a value. Every transport-specific key defaults to None or empty."""
    return bool(getattr(model, key, None))


TRANSPORT_KEYS: dict[str, set[str]] = {
    "stdio": {"command", "cwd", "env"},
    "http": {"url", "auth_token_env", "headers"},
    "in_memory": set(),
}
"""Which optional keys each transport accepts (PRD §5b "Transport modes", §13).

Validated per server rather than merely tolerated: the migration people will actually perform is
stdio → in_memory, and a `command:` left behind on a switched server would otherwise sit there
looking meaningful while nothing read it.
"""


class McpServerConfig(StrictModel):
    """One MCP tool source for the worker agent (PRD §5b).

    Three transports, differing in where the server runs and therefore in what has to be
    configured here: `in_memory` needs nothing but a `name` (the server ships with this package
    and its credentials come from the environment), while `stdio` and `http` carry their whole
    description in this file.
    """

    name: str = Field(
        description="Identifies the server in citations (`Source.server`), metrics, error messages, "
        "and `--mcp-args`. For `in_memory`, also selects which bundled server to load."
    )
    transport: Literal["stdio", "http", "in_memory"] = "stdio"

    # --- stdio only
    command: list[str] | None = Field(
        default=None, description="argv of the subprocess to launch, e.g. ['uv', 'run', 'python', '-m', 'zotero_mcp']."
    )
    cwd: str | None = Field(
        default=None,
        description="Working directory for the subprocess. Relative paths resolve against the "
        "PROJECT folder, never the install location (PRD §4a).",
    )
    env: dict[str, str] = Field(
        default_factory=dict,
        description="Environment for the subprocess. Values may use ${VAR} / ${VAR:-default} "
        "references, expanded at startup — so a secret is named here, never stored here.",
    )

    # --- http only
    url: str | None = Field(default=None, description="Streamable-HTTP endpoint of a remote server.")
    auth_token_env: str | None = Field(
        default=None,
        description="NAME of the environment variable holding the bearer token — not the token. A "
        "literal token in config.yaml is a token in someone's git history.",
    )
    headers: dict[str, str] | None = Field(default=None, description="Extra HTTP headers; ${VAR} references allowed.")

    # --- every transport
    optional: bool = Field(
        default=False,
        description="False (default): a server that fails to start aborts the run, because a "
        "silently web-only report looks complete and isn't (PRD §5b). True: warn, drop the server, "
        "and record the degradation in the report's assumptions.",
    )
    timeout_seconds: float = Field(default=60.0, gt=0, description="Per-tool-call timeout.")
    max_tool_retries: int = Field(
        default=3,
        ge=1,
        description="How many times the model may retry ONE tool after a validation error the "
        "server hands back (Pydantic AI's ModelRetry). The library default of 1 gives the model a "
        "single shot at acting on the server's corrective message; a second mistake on the same "
        "tool kills the whole worker run — which real Zotero runs hit constantly. The counter "
        "resets on a successful call, so this bounds consecutive failures, not total use.",
    )
    allowed_tools: list[str] | None = Field(
        default=None,
        description="Allowlist applied after the server's tool list is fetched. None exposes every "
        "tool the server offers. A name the server doesn't have is a startup error.",
    )
    health_check: str | None = Field(
        default=None,
        description="A cheap read-only tool the preflight calls to prove the server actually works. "
        "Listing tools is not proof: zotero-mcp accepts a rejected API key at startup by design, so "
        "a list-only probe would pass and then every call would fail (PRD §5b).",
    )
    tool_args: dict[str, dict[str, Any]] = Field(
        default_factory=dict,
        description="Per-tool arguments merged into every matching call before dispatch. Wins over "
        "model-supplied values for the same key — the point is a floor the model cannot lower.",
    )
    instructions: str | None = Field(
        default=None,
        description="Free-text hint appended to the worker prompt: what this server is for and when "
        "to prefer it. Injection alone leaves the worker not knowing when to use the server.",
    )

    @model_validator(mode="after")
    def _check_transport_keys(self) -> McpServerConfig:
        legal = TRANSPORT_KEYS[self.transport]
        foreign = sorted(
            key
            for other, keys in TRANSPORT_KEYS.items()
            if other != self.transport
            for key in keys - legal
            if _is_set(self, key)
        )
        if foreign:
            raise ValueError(
                f"MCP server {self.name!r} uses transport {self.transport!r}, which does not accept "
                f"{', '.join(foreign)}. Remove the key, or change the transport."
            )
        if self.transport == "stdio" and not self.command:
            raise ValueError(f"MCP server {self.name!r} (stdio) needs a `command` to launch.")
        if self.transport == "http" and not self.url:
            raise ValueError(f"MCP server {self.name!r} (http) needs a `url` to connect to.")
        return self

    def referenced_tools(self) -> set[str]:
        """Every tool name this config names, for validation against the server's real tool list."""
        named = set(self.tool_args) | set(self.allowed_tools or ())
        if self.health_check:
            named.add(self.health_check)
        return named


class BudgetsConfig(StrictModel):
    breadth_budget: int = 8
    depth_budget: int = 2
    worker_concurrency: int = 4
    spend_limit_usd: float = 5.0
    total_tokens_limit: int = 1_500_000

    request_limit: int = Field(
        default=300,
        gt=0,
        description="Max model requests for RESEARCH in one run. Every agent call in the run "
        "shares one accumulator, so this bounds the whole research phase, not each call. Left "
        "unset, Pydantic AI applies its own default of 50 - far below what one round of workers "
        "needs, which made it the first ceiling a real run hit even with tokens and USD to spare.",
    )
    report_request_allowance: int = Field(
        default=30,
        gt=0,
        description="Requests guaranteed to synthesis+critique, granted on top of however many "
        "research used - the same 'on top of actual usage' guarantee as the token and USD "
        "allowances. Without it, a run that spent its request ceiling on research could not "
        "issue even one request to write the report.",
    )

    report_token_allowance: int = Field(
        default=250_000,
        gt=0,
        description="Tokens guaranteed to synthesis+critique, granted on top of whatever research "
        "actually spent rather than carved out of the run ceiling. Sized so a report over a full "
        "breadth of findings, plus a critique of it, fits comfortably.",
    )
    report_spend_allowance_usd: float = Field(
        default=1.0,
        gt=0,
        description="The same guarantee in USD, for models with pricing data. Both apply; "
        "whichever binds first stops the write-up.",
    )


class OutputConfig(StrictModel):
    report_dir: str = "reports"
    state_dir: str = ".deep_research"


class LoggingConfig(StrictModel):
    logfire: bool = True


class AppConfig(StrictModel):
    model: ModelConfig
    prices: dict[str, ModelPrice] = Field(
        default_factory=dict,
        description="Fallback USD rates keyed by the exact model string used in `model:`. Applied "
        "only where genai-prices has no data, so a priced model is never second-guessed by a "
        "hand-entered number.",
    )
    search: SearchConfig = Field(default_factory=SearchConfig)
    mcp_servers: list[McpServerConfig] = Field(default_factory=list)
    budgets: BudgetsConfig = Field(default_factory=BudgetsConfig)
    output: OutputConfig = Field(default_factory=OutputConfig)
    logging: LoggingConfig = Field(default_factory=LoggingConfig)

    @model_validator(mode="after")
    def _check_server_names_unique(self) -> AppConfig:
        seen: set[str] = set()
        for server in self.mcp_servers:
            if server.name in seen:
                # `Source.server`, the metrics ledger, and `--mcp-args` all key on this name.
                raise ValueError(f"Duplicate MCP server name {server.name!r} — names must be unique.")
            seen.add(server.name)
        return self

    def mcp_server(self, name: str) -> McpServerConfig | None:
        return next((s for s in self.mcp_servers if s.name == name), None)

    def report_dir(self, project_dir: Path) -> Path:
        """Resolve `output.report_dir` against the project folder, not the install location."""
        path = Path(self.output.report_dir).expanduser()
        if not path.is_absolute():
            path = project_dir / path
        return path

    def state_dir(self, project_dir: Path) -> Path:
        """Resolve `output.state_dir` — where `ResearchState` checkpoints (PRD §7) are saved."""
        path = Path(self.output.state_dir).expanduser()
        if not path.is_absolute():
            path = project_dir / path
        return path


def apply_model_env_overrides(config: AppConfig) -> AppConfig:
    """Let the environment override the configured models, in place.

    `DEEP_RESEARCH_MODEL` sets all three roles; the per-role variables win over it. This is what
    makes a one-off run against a different provider a prefix rather than a file edit:

        DEEP_RESEARCH_MODEL=openai:gpt-5-mini deep-research run --auto "..."
    """
    shared = os.environ.get(MODEL_ENV_VAR)
    for role, env_var in ROLE_MODEL_ENV_VARS.items():
        chosen = os.environ.get(env_var) or shared
        if chosen:
            setattr(config.model, role, chosen)
    return config


def bundled_default_config_text() -> str:
    """The tool's built-in default config, used when a project has no `config.yaml` yet."""
    return resources.files("deep_research").joinpath("default_config.yaml").read_text()


def find_project_config(project_dir: Path) -> Path | None:
    candidate = project_dir / PROJECT_CONFIG_NAME
    return candidate if candidate.is_file() else None


def load_config(path: str | Path | None = None, *, project_dir: Path | None = None) -> AppConfig:
    """Load config per the lookup order in the module docstring.

    `project_dir` defaults to the current working directory — the researcher's project folder.
    """
    project_dir = project_dir if project_dir is not None else Path.cwd()

    if path is not None:
        source = Path(path)
        raw_text = source.read_text()
    else:
        project_config = find_project_config(project_dir)
        source = project_config if project_config is not None else Path("<bundled default_config.yaml>")
        raw_text = project_config.read_text() if project_config is not None else bundled_default_config_text()

    raw = yaml.safe_load(raw_text) or {}
    try:
        return apply_model_env_overrides(AppConfig.model_validate(raw))
    except ValidationError as exc:
        # Name the file. A validation error against an anonymous dict is a poor thing to hand
        # someone who mistyped a key three directories away from where they ran the command.
        raise ValueError(f"{source} is not a valid deep-research config:\n{exc}") from exc
