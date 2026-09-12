"""Alembic environment for DocSentry.

The database URL is never written in alembic.ini. It comes from the same
`DATABASE_URL` the application uses, through `app.db.database_url`, so a
migration can only ever run against the database the app would talk to. A
connection string duplicated into a config file is a connection string that
eventually points somewhere else.

There are no SQLAlchemy models here on purpose. The vector table is created and
owned by langchain-postgres, and the `events` table is written through plain
SQL. Autogenerate against a metadata object that describes neither would offer
to drop both, so revisions are written by hand.
"""

import os
import sys
from logging.config import fileConfig

from alembic import context
from sqlalchemy import engine_from_config, pool

# The app package lives one level up from this directory.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv  # noqa: E402

# Alembic runs outside the app process, so the .env the app relies on is not
# loaded for it. Without this a local `alembic upgrade head` silently falls back
# to the default connection string instead of the one in .env.
load_dotenv(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), ".env"))

from app.db import database_url  # noqa: E402

config = context.config

if config.config_file_name is not None:
    fileConfig(config.config_file_name)

config.set_main_option("sqlalchemy.url", database_url())

target_metadata = None


def run_migrations_offline() -> None:
    """Emit SQL to stdout instead of running it.

    Useful for Supabase, where a migration can be reviewed and pasted into the
    SQL editor rather than run over a connection.
    """
    context.configure(
        url=database_url(),
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
    )
    with context.begin_transaction():
        context.run_migrations()


def run_migrations_online() -> None:
    connectable = engine_from_config(
        config.get_section(config.config_ini_section, {}),
        prefix="sqlalchemy.",
        poolclass=pool.NullPool,
    )
    with connectable.connect() as connection:
        context.configure(connection=connection, target_metadata=target_metadata)
        with context.begin_transaction():
            context.run_migrations()


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
