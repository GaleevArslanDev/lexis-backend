"""add ml api v1 fields

Revision ID: add_ml_api_v1_fields
Revises: add_assessit_models
Create Date: 2025-12-13 10:00:00.000000

"""
from typing import Sequence, Union
from alembic import op
import sqlalchemy as sa
import sqlmodel

# revision identifiers, used by Alembic.
revision: str = 'add_ml_api_v1_fields'
down_revision: Union[str, Sequence[str], None] = 'add_assessit_models'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # Добавляем новые поля в recognizedsolution
    op.add_column('recognizedsolution', sa.Column('solution_id', sqlmodel.sql.sqltypes.AutoString(), nullable=True))
    op.add_column('recognizedsolution', sa.Column('mark_score', sa.Float(), nullable=True))
    op.add_column('recognizedsolution', sa.Column('teacher_comment', sqlmodel.sql.sqltypes.AutoString(), nullable=True))
    op.add_column('recognizedsolution', sa.Column('c_ocr', sa.Float(), nullable=True))
    op.add_column('recognizedsolution', sa.Column('c_llm', sa.Float(), nullable=True))
    op.add_column('recognizedsolution', sa.Column('m_sympy', sa.Float(), nullable=True))
    op.add_column('recognizedsolution', sa.Column('m_llm', sa.Float(), nullable=True))
    op.add_column('recognizedsolution', sa.Column('m_answer', sa.Float(), nullable=True))
    op.add_column('recognizedsolution', sa.Column('steps_analysis_json', sqlmodel.sql.sqltypes.AutoString(), nullable=True))
    op.add_column('recognizedsolution', sa.Column('api_version', sqlmodel.sql.sqltypes.AutoString(), nullable=True))
    op.add_column('recognizedsolution', sa.Column('job_id', sqlmodel.sql.sqltypes.AutoString(), nullable=True))

    # Создаем индексы
    op.create_index(op.f('ix_recognizedsolution_solution_id'), 'recognizedsolution', ['solution_id'], unique=False)
    op.create_index(op.f('ix_recognizedsolution_job_id'), 'recognizedsolution', ['job_id'], unique=False)


def downgrade() -> None:
    # Удаляем индексы
    op.drop_index(op.f('ix_recognizedsolution_job_id'), table_name='recognizedsolution')
    op.drop_index(op.f('ix_recognizedsolution_solution_id'), table_name='recognizedsolution')

    # Удаляем поля
    op.drop_column('recognizedsolution', 'job_id')
    op.drop_column('recognizedsolution', 'api_version')
    op.drop_column('recognizedsolution', 'steps_analysis_json')
    op.drop_column('recognizedsolution', 'm_answer')
    op.drop_column('recognizedsolution', 'm_llm')
    op.drop_column('recognizedsolution', 'm_sympy')
    op.drop_column('recognizedsolution', 'c_llm')
    op.drop_column('recognizedsolution', 'c_ocr')
    op.drop_column('recognizedsolution', 'teacher_comment')
    op.drop_column('recognizedsolution', 'mark_score')
    op.drop_column('recognizedsolution', 'solution_id')