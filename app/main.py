from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI

from app.agent_service import InterviewAgentService
from app.api import router
from app.config import Settings, get_settings
from app.database import Database
from app.model_gateway import ModelGateway, build_gateway


def create_app(
    settings: Settings | None = None,
    gateway: ModelGateway | None = None,
) -> FastAPI:
    resolved_settings = settings or get_settings()
    resolved_gateway = gateway or build_gateway(resolved_settings)
    database = Database(resolved_settings.database_url)

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        # Automatic schema creation keeps the P0 demo self-contained.
        # Production deployments should replace this with versioned migrations.
        await database.create_schema()
        yield
        await resolved_gateway.close()
        await database.dispose()

    app = FastAPI(
        title=resolved_settings.app_name,
        version="0.1.0",
        lifespan=lifespan,
    )
    app.state.settings = resolved_settings
    app.state.database = database
    app.state.agent_service = InterviewAgentService(
        resolved_gateway,
        resolved_settings,
    )
    app.include_router(router)
    return app


app = create_app()
