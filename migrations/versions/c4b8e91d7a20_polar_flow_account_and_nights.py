"""polar flow account and synced nights

Revision ID: c4b8e91d7a20
Revises: e7a41c8d5f02
Create Date: 2026-08-21 10:00:00.000000

"""
from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision = 'c4b8e91d7a20'
down_revision = 'e7a41c8d5f02'
branch_labels = None
depends_on = None


def upgrade():
    op.create_table(
        'polar_account',
        sa.Column('id', sa.Integer(), nullable=False),
        sa.Column('person_id', sa.Integer(), nullable=False),
        sa.Column('polar_user_id', sa.String(length=64), nullable=False),
        sa.Column('access_token', sa.String(length=255), nullable=False),
        sa.Column('member_id', sa.String(length=120), nullable=False),
        sa.Column('linked_at', sa.DateTime(), nullable=False),
        sa.Column('last_sync_at', sa.DateTime(), nullable=True),
        sa.Column('last_sync_note', sa.String(length=500), nullable=True),
        sa.ForeignKeyConstraint(['person_id'], ['person.id']),
        sa.PrimaryKeyConstraint('id'),
    )
    with op.batch_alter_table('polar_account', schema=None) as batch_op:
        batch_op.create_index(
            batch_op.f('ix_polar_account_person_id'), ['person_id'], unique=True
        )
        batch_op.create_index(
            batch_op.f('ix_polar_account_polar_user_id'),
            ['polar_user_id'],
            unique=False,
        )

    op.create_table(
        'flow_night',
        sa.Column('id', sa.Integer(), nullable=False),
        sa.Column('person_id', sa.Integer(), nullable=False),
        sa.Column('date', sa.Date(), nullable=False),
        sa.Column('source_device_id', sa.String(length=64), nullable=True),
        # Sleep Plus Stages
        sa.Column('sleep_start', sa.DateTime(), nullable=True),
        sa.Column('sleep_end', sa.DateTime(), nullable=True),
        sa.Column('light_sleep_s', sa.Integer(), nullable=True),
        sa.Column('deep_sleep_s', sa.Integer(), nullable=True),
        sa.Column('rem_sleep_s', sa.Integer(), nullable=True),
        sa.Column('unrecognized_sleep_s', sa.Integer(), nullable=True),
        sa.Column('total_interruption_s', sa.Integer(), nullable=True),
        sa.Column('sleep_score', sa.Integer(), nullable=True),
        sa.Column('sleep_charge', sa.Integer(), nullable=True),
        sa.Column('continuity', sa.Float(), nullable=True),
        sa.Column('continuity_class', sa.Integer(), nullable=True),
        sa.Column('sleep_cycles', sa.Integer(), nullable=True),
        # Nightly Recharge
        sa.Column('hr_avg_bpm', sa.Integer(), nullable=True),
        sa.Column('beat_to_beat_avg_ms', sa.Integer(), nullable=True),
        sa.Column('hrv_rmssd_ms', sa.Integer(), nullable=True),
        sa.Column('breathing_rate_avg', sa.Float(), nullable=True),
        sa.Column('nightly_recharge_status', sa.Integer(), nullable=True),
        sa.Column('ans_charge', sa.Float(), nullable=True),
        sa.Column('ans_charge_status', sa.Integer(), nullable=True),
        # Small display-only series
        sa.Column('hrv_samples', sa.JSON(), nullable=True),
        sa.Column('breathing_samples', sa.JSON(), nullable=True),
        sa.Column('hr_samples', sa.JSON(), nullable=True),
        sa.Column('fetched_at', sa.DateTime(), nullable=False),
        sa.ForeignKeyConstraint(['person_id'], ['person.id']),
        sa.PrimaryKeyConstraint('id'),
        sa.UniqueConstraint('person_id', 'date', name='uq_flow_night_person_date'),
    )
    with op.batch_alter_table('flow_night', schema=None) as batch_op:
        batch_op.create_index(
            batch_op.f('ix_flow_night_person_id'), ['person_id'], unique=False
        )
        batch_op.create_index(batch_op.f('ix_flow_night_date'), ['date'], unique=False)


def downgrade():
    with op.batch_alter_table('flow_night', schema=None) as batch_op:
        batch_op.drop_index(batch_op.f('ix_flow_night_date'))
        batch_op.drop_index(batch_op.f('ix_flow_night_person_id'))
    op.drop_table('flow_night')

    with op.batch_alter_table('polar_account', schema=None) as batch_op:
        batch_op.drop_index(batch_op.f('ix_polar_account_polar_user_id'))
        batch_op.drop_index(batch_op.f('ix_polar_account_person_id'))
    op.drop_table('polar_account')
