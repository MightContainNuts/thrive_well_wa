import sqlmodel

"""renamed column

Revision ID: 658eb61099e6
Revises: 0c03901a0fed
Create Date: 2025-04-16 13:44:35.403020

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = "658eb61099e6"
down_revision: Union[str, None] = "0c03901a0fed"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Upgrade schema."""
    # ### commands auto generated by Alembic - please adjust! ###
    op.drop_column("message", "msg_text")
    op.drop_column("user", "last_name")
    op.drop_column("user", "first_name")
    # ### end Alembic commands ###


def downgrade() -> None:
    """Downgrade schema."""
    # ### commands auto generated by Alembic - please adjust! ###
    op.add_column(
        "user",
        sa.Column("first_name", sa.VARCHAR(), autoincrement=False, nullable=False),
    )
    op.add_column(
        "user",
        sa.Column("last_name", sa.VARCHAR(), autoincrement=False, nullable=False),
    )
    op.add_column(
        "message",
        sa.Column("msg_text", sa.VARCHAR(), autoincrement=False, nullable=False),
    )
    # ### end Alembic commands ###
