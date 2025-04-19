import sqlmodel

"""revamped columns for large int

Revision ID: c7b3e89ca7d4
Revises: 3acc3b14c0b6
Create Date: 2025-04-19 15:30:32.505096

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = "c7b3e89ca7d4"
down_revision: Union[str, None] = "3acc3b14c0b6"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Upgrade schema."""
    # ### commands auto generated by Alembic - please adjust! ###
    op.alter_column(
        "message",
        "timestamp",
        existing_type=sa.BIGINT(),
        type_=sa.Integer(),
        existing_nullable=False,
    )
    # ### end Alembic commands ###


def downgrade() -> None:
    """Downgrade schema."""
    # ### commands auto generated by Alembic - please adjust! ###
    op.alter_column(
        "message",
        "timestamp",
        existing_type=sa.Integer(),
        type_=sa.BIGINT(),
        existing_nullable=False,
    )
    # ### end Alembic commands ###
