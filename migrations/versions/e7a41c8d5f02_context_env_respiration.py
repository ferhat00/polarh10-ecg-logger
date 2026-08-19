"""structured context fields, environment capture, ECG-derived respiration

Revision ID: e7a41c8d5f02
Revises: b3d1f0a2c9e4
Create Date: 2026-08-19 12:00:00.000000

"""
from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision = 'e7a41c8d5f02'
down_revision = 'b3d1f0a2c9e4'
branch_labels = None
depends_on = None


def upgrade():
    with op.batch_alter_table('session', schema=None) as batch_op:
        # Structured context (all nullable — absence is the normal case)
        batch_op.add_column(sa.Column('body_position', sa.String(length=12), nullable=True))
        batch_op.add_column(
            sa.Column('body_position_source', sa.String(length=10), nullable=True)
        )
        batch_op.add_column(sa.Column('alcohol_drinks_24h', sa.Integer(), nullable=True))
        batch_op.add_column(sa.Column('sleep_quality_1_5', sa.Integer(), nullable=True))
        batch_op.create_check_constraint(
            'ck_session_body_position',
            "body_position IN ('supine', 'prone', 'left', 'right', "
            "'sitting', 'standing', 'moving', 'unknown')",
        )
        # Environment at recording time (opt-in Open-Meteo lookup)
        batch_op.add_column(sa.Column('env_temp_c', sa.Float(), nullable=True))
        batch_op.add_column(sa.Column('env_apparent_temp_c', sa.Float(), nullable=True))
        batch_op.add_column(sa.Column('env_humidity_pct', sa.Float(), nullable=True))
        batch_op.add_column(sa.Column('env_pressure_hpa', sa.Float(), nullable=True))
        batch_op.add_column(sa.Column('env_pm25_ugm3', sa.Float(), nullable=True))
        batch_op.add_column(sa.Column('env_pm10_ugm3', sa.Float(), nullable=True))
        batch_op.add_column(sa.Column('env_ozone_ugm3', sa.Float(), nullable=True))
        batch_op.add_column(sa.Column('env_no2_ugm3', sa.Float(), nullable=True))
        batch_op.add_column(sa.Column('env_aqi', sa.Float(), nullable=True))
        batch_op.add_column(sa.Column('env_daylight_h', sa.Float(), nullable=True))
        batch_op.add_column(sa.Column('env_source', sa.String(length=24), nullable=True))
        batch_op.add_column(sa.Column('env_fetched_at', sa.DateTime(), nullable=True))

    with op.batch_alter_table('metrics', schema=None) as batch_op:
        batch_op.add_column(sa.Column('resp_rate_median_brpm', sa.Float(), nullable=True))
        batch_op.add_column(sa.Column('resp_rate_p5_brpm', sa.Float(), nullable=True))
        batch_op.add_column(sa.Column('resp_rate_p95_brpm', sa.Float(), nullable=True))


def downgrade():
    with op.batch_alter_table('metrics', schema=None) as batch_op:
        batch_op.drop_column('resp_rate_p95_brpm')
        batch_op.drop_column('resp_rate_p5_brpm')
        batch_op.drop_column('resp_rate_median_brpm')

    with op.batch_alter_table('session', schema=None) as batch_op:
        batch_op.drop_column('env_fetched_at')
        batch_op.drop_column('env_source')
        batch_op.drop_column('env_daylight_h')
        batch_op.drop_column('env_aqi')
        batch_op.drop_column('env_no2_ugm3')
        batch_op.drop_column('env_ozone_ugm3')
        batch_op.drop_column('env_pm10_ugm3')
        batch_op.drop_column('env_pm25_ugm3')
        batch_op.drop_column('env_pressure_hpa')
        batch_op.drop_column('env_humidity_pct')
        batch_op.drop_column('env_apparent_temp_c')
        batch_op.drop_column('env_temp_c')
        batch_op.drop_constraint('ck_session_body_position', type_='check')
        batch_op.drop_column('sleep_quality_1_5')
        batch_op.drop_column('alcohol_drinks_24h')
        batch_op.drop_column('body_position_source')
        batch_op.drop_column('body_position')
