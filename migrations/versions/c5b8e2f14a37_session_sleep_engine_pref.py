"""session sleep_engine_pref

Revision ID: c5b8e2f14a37
Revises: e7a41c8d5f02
Create Date: 2026-08-21 10:00:00.000000

"""
from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision = 'c5b8e2f14a37'
down_revision = 'e7a41c8d5f02'
branch_labels = None
depends_on = None


def upgrade():
    with op.batch_alter_table('session', schema=None) as batch_op:
        batch_op.add_column(
            sa.Column('sleep_engine_pref', sa.String(length=40), nullable=True)
        )


def downgrade():
    with op.batch_alter_table('session', schema=None) as batch_op:
        batch_op.drop_column('sleep_engine_pref')
