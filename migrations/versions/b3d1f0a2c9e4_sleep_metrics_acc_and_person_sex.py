"""sleep metrics, ACC file columns, person sex

Revision ID: b3d1f0a2c9e4
Revises: 9c41d2aa7b05
Create Date: 2026-08-18 18:30:00.000000

"""
from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision = 'b3d1f0a2c9e4'
down_revision = '9c41d2aa7b05'
branch_labels = None
depends_on = None


def upgrade():
    with op.batch_alter_table('person', schema=None) as batch_op:
        batch_op.add_column(sa.Column('sex', sa.String(length=10), nullable=True))

    with op.batch_alter_table('session', schema=None) as batch_op:
        batch_op.add_column(
            sa.Column('acc_original_filename', sa.String(length=255), nullable=True)
        )
        batch_op.add_column(
            sa.Column('acc_stored_path', sa.String(length=500), nullable=True)
        )
        batch_op.add_column(
            sa.Column('acc_file_sha256', sa.String(length=64), nullable=True)
        )
        batch_op.create_index(
            batch_op.f('ix_session_acc_file_sha256'), ['acc_file_sha256'], unique=False
        )

    with op.batch_alter_table('metrics', schema=None) as batch_op:
        batch_op.add_column(sa.Column('tst_min', sa.Float(), nullable=True))
        batch_op.add_column(sa.Column('sleep_efficiency_pct', sa.Float(), nullable=True))
        batch_op.add_column(sa.Column('sol_min', sa.Float(), nullable=True))
        batch_op.add_column(sa.Column('waso_min', sa.Float(), nullable=True))
        batch_op.add_column(sa.Column('light_min', sa.Float(), nullable=True))
        batch_op.add_column(sa.Column('deep_min', sa.Float(), nullable=True))
        batch_op.add_column(sa.Column('rem_min', sa.Float(), nullable=True))
        batch_op.add_column(sa.Column('awakenings_n', sa.Integer(), nullable=True))
        batch_op.add_column(sa.Column('sleep_engine', sa.String(length=40), nullable=True))


def downgrade():
    with op.batch_alter_table('metrics', schema=None) as batch_op:
        batch_op.drop_column('sleep_engine')
        batch_op.drop_column('awakenings_n')
        batch_op.drop_column('rem_min')
        batch_op.drop_column('deep_min')
        batch_op.drop_column('light_min')
        batch_op.drop_column('waso_min')
        batch_op.drop_column('sol_min')
        batch_op.drop_column('sleep_efficiency_pct')
        batch_op.drop_column('tst_min')

    with op.batch_alter_table('session', schema=None) as batch_op:
        batch_op.drop_index(batch_op.f('ix_session_acc_file_sha256'))
        batch_op.drop_column('acc_file_sha256')
        batch_op.drop_column('acc_stored_path')
        batch_op.drop_column('acc_original_filename')

    with op.batch_alter_table('person', schema=None) as batch_op:
        batch_op.drop_column('sex')
