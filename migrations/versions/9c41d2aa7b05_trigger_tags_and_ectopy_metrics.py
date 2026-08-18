"""trigger tags and ectopy metrics

Revision ID: 9c41d2aa7b05
Revises: 7ec56b135b83
Create Date: 2026-08-18 05:30:00.000000

"""
from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision = '9c41d2aa7b05'
down_revision = '7ec56b135b83'
branch_labels = None
depends_on = None


def upgrade():
    op.create_table(
        'trigger_tag',
        sa.Column('id', sa.Integer(), nullable=False),
        sa.Column('name', sa.String(length=80), nullable=False),
        sa.Column('slug', sa.String(length=100), nullable=False),
        sa.Column('is_builtin', sa.Boolean(), nullable=False),
        sa.Column('created_at', sa.DateTime(), nullable=False),
        sa.PrimaryKeyConstraint('id'),
        sa.UniqueConstraint('name'),
        sa.UniqueConstraint('slug'),
    )
    with op.batch_alter_table('trigger_tag', schema=None) as batch_op:
        batch_op.create_index(batch_op.f('ix_trigger_tag_slug'), ['slug'], unique=True)

    op.create_table(
        'session_trigger_tag',
        sa.Column('session_id', sa.Integer(), nullable=False),
        sa.Column('trigger_tag_id', sa.Integer(), nullable=False),
        sa.ForeignKeyConstraint(['session_id'], ['session.id']),
        sa.ForeignKeyConstraint(['trigger_tag_id'], ['trigger_tag.id']),
        sa.PrimaryKeyConstraint('session_id', 'trigger_tag_id'),
    )

    with op.batch_alter_table('metrics', schema=None) as batch_op:
        batch_op.add_column(sa.Column('ectopy_beats_n', sa.Integer(), nullable=True))
        batch_op.add_column(sa.Column('ectopy_per_hour', sa.Float(), nullable=True))
        batch_op.add_column(sa.Column('ectopy_pct_beats', sa.Float(), nullable=True))
        batch_op.add_column(sa.Column('single_n', sa.Integer(), nullable=True))
        batch_op.add_column(sa.Column('couplet_n', sa.Integer(), nullable=True))
        batch_op.add_column(sa.Column('run_n', sa.Integer(), nullable=True))
        batch_op.add_column(sa.Column('longest_run_beats', sa.Integer(), nullable=True))
        batch_op.add_column(sa.Column('bigeminy_episode_n', sa.Integer(), nullable=True))
        batch_op.add_column(sa.Column('trigeminy_episode_n', sa.Integer(), nullable=True))


def downgrade():
    with op.batch_alter_table('metrics', schema=None) as batch_op:
        batch_op.drop_column('trigeminy_episode_n')
        batch_op.drop_column('bigeminy_episode_n')
        batch_op.drop_column('longest_run_beats')
        batch_op.drop_column('run_n')
        batch_op.drop_column('couplet_n')
        batch_op.drop_column('single_n')
        batch_op.drop_column('ectopy_pct_beats')
        batch_op.drop_column('ectopy_per_hour')
        batch_op.drop_column('ectopy_beats_n')

    op.drop_table('session_trigger_tag')
    with op.batch_alter_table('trigger_tag', schema=None) as batch_op:
        batch_op.drop_index(batch_op.f('ix_trigger_tag_slug'))
    op.drop_table('trigger_tag')
