BEGIN;

CREATE TABLE alembic_version (
    version_num VARCHAR(32) NOT NULL, 
    CONSTRAINT alembic_version_pkc PRIMARY KEY (version_num)
);

-- Running upgrade  -> 5d25d7274339

CREATE TABLE actionlog (
    id SERIAL NOT NULL, 
    user_id INTEGER NOT NULL, 
    action_type VARCHAR NOT NULL, 
    target_type VARCHAR NOT NULL, 
    target_id INTEGER NOT NULL, 
    timestamp TIMESTAMP WITHOUT TIME ZONE NOT NULL, 
    details VARCHAR, 
    PRIMARY KEY (id)
);

CREATE TABLE result (
    id SERIAL NOT NULL, 
    student_id INTEGER NOT NULL, 
    assignment_id INTEGER NOT NULL, 
    total_score FLOAT, 
    feedback_summary VARCHAR, 
    PRIMARY KEY (id)
);

CREATE TABLE studentanswer (
    id SERIAL NOT NULL, 
    student_id INTEGER, 
    question_id INTEGER, 
    text_answer VARCHAR, 
    file_url VARCHAR, 
    score FLOAT, 
    feedback VARCHAR, 
    created_at TIMESTAMP WITHOUT TIME ZONE NOT NULL, 
    PRIMARY KEY (id)
);

CREATE TABLE "user" (
    id SERIAL NOT NULL, 
    email VARCHAR NOT NULL, 
    hashed_password VARCHAR NOT NULL, 
    role VARCHAR NOT NULL, 
    created_at TIMESTAMP WITHOUT TIME ZONE NOT NULL, 
    PRIMARY KEY (id)
);

CREATE UNIQUE INDEX ix_user_email ON "user" (email);

CREATE TABLE school (
    id SERIAL NOT NULL, 
    name VARCHAR NOT NULL, 
    address VARCHAR, 
    creator_id INTEGER NOT NULL, 
    created_at TIMESTAMP WITHOUT TIME ZONE NOT NULL, 
    PRIMARY KEY (id), 
    FOREIGN KEY(creator_id) REFERENCES "user" (id)
);

CREATE UNIQUE INDEX ix_school_name ON school (name);

CREATE TABLE class (
    id SERIAL NOT NULL, 
    name VARCHAR NOT NULL, 
    teacher_id INTEGER NOT NULL, 
    school_id INTEGER, 
    created_at TIMESTAMP WITHOUT TIME ZONE NOT NULL, 
    PRIMARY KEY (id), 
    FOREIGN KEY(school_id) REFERENCES school (id), 
    FOREIGN KEY(teacher_id) REFERENCES "user" (id)
);

CREATE TABLE assignment (
    id SERIAL NOT NULL, 
    class_id INTEGER NOT NULL, 
    title VARCHAR NOT NULL, 
    description VARCHAR, 
    type VARCHAR NOT NULL, 
    created_at TIMESTAMP WITHOUT TIME ZONE NOT NULL, 
    due_date TIMESTAMP WITHOUT TIME ZONE, 
    PRIMARY KEY (id), 
    FOREIGN KEY(class_id) REFERENCES class (id)
);

CREATE TABLE classstudentlink (
    class_id INTEGER NOT NULL, 
    student_id INTEGER NOT NULL, 
    PRIMARY KEY (class_id, student_id), 
    FOREIGN KEY(class_id) REFERENCES class (id), 
    FOREIGN KEY(student_id) REFERENCES "user" (id)
);

CREATE TABLE question (
    id SERIAL NOT NULL, 
    assignment_id INTEGER NOT NULL, 
    text VARCHAR NOT NULL, 
    options VARCHAR, 
    correct_answer VARCHAR, 
    rubric_criteria VARCHAR, 
    PRIMARY KEY (id), 
    FOREIGN KEY(assignment_id) REFERENCES assignment (id)
);

CREATE TABLE rubric (
    id SERIAL NOT NULL, 
    assignment_id INTEGER NOT NULL, 
    criteria_json VARCHAR, 
    max_score FLOAT, 
    PRIMARY KEY (id), 
    FOREIGN KEY(assignment_id) REFERENCES assignment (id)
);

CREATE TABLE mistakestat (
    id SERIAL NOT NULL, 
    class_id INTEGER NOT NULL, 
    question_id INTEGER NOT NULL, 
    mistake_type VARCHAR NOT NULL, 
    count INTEGER NOT NULL, 
    created_at TIMESTAMP WITHOUT TIME ZONE NOT NULL, 
    PRIMARY KEY (id), 
    FOREIGN KEY(class_id) REFERENCES class (id), 
    FOREIGN KEY(question_id) REFERENCES question (id)
);

INSERT INTO alembic_version (version_num) VALUES ('5d25d7274339') RETURNING alembic_version.version_num;

-- Running upgrade 5d25d7274339 -> add_name_surname_to_user

ALTER TABLE "user" ADD COLUMN name VARCHAR DEFAULT '' NOT NULL;

ALTER TABLE "user" ADD COLUMN surname VARCHAR DEFAULT '' NOT NULL;

UPDATE alembic_version SET version_num='add_name_surname_to_user' WHERE alembic_version.version_num = '5d25d7274339';

-- Running upgrade add_name_surname_to_user -> add_assessit_models

ALTER TABLE assignment ADD COLUMN reference_solution VARCHAR;

ALTER TABLE assignment ADD COLUMN reference_answer VARCHAR;

ALTER TABLE assignment ADD COLUMN subject VARCHAR;

ALTER TABLE assignment ADD COLUMN difficulty VARCHAR;

ALTER TABLE assignment ADD COLUMN max_score FLOAT;

ALTER TABLE question ADD COLUMN formula_template VARCHAR;

ALTER TABLE question ADD COLUMN step_by_step_solution VARCHAR;

ALTER TABLE question ADD COLUMN common_mistakes VARCHAR;

CREATE TABLE assessmentimage (
    id SERIAL NOT NULL, 
    assignment_id INTEGER NOT NULL, 
    class_id INTEGER, 
    student_id INTEGER NOT NULL, 
    question_id INTEGER, 
    original_image_path VARCHAR NOT NULL, 
    processed_image_path VARCHAR, 
    thumbnail_path VARCHAR, 
    file_name VARCHAR NOT NULL, 
    file_size INTEGER NOT NULL, 
    mime_type VARCHAR NOT NULL, 
    upload_time TIMESTAMP WITHOUT TIME ZONE NOT NULL, 
    status VARCHAR NOT NULL, 
    processing_started TIMESTAMP WITHOUT TIME ZONE, 
    processing_completed TIMESTAMP WITHOUT TIME ZONE, 
    error_message VARCHAR, 
    retry_count INTEGER NOT NULL, 
    PRIMARY KEY (id), 
    FOREIGN KEY(assignment_id) REFERENCES assignment (id), 
    FOREIGN KEY(class_id) REFERENCES class (id), 
    FOREIGN KEY(student_id) REFERENCES "user" (id), 
    FOREIGN KEY(question_id) REFERENCES question (id)
);

CREATE TABLE recognizedsolution (
    id SERIAL NOT NULL, 
    image_id INTEGER NOT NULL, 
    extracted_text VARCHAR NOT NULL, 
    cleaned_text VARCHAR, 
    text_confidence FLOAT, 
    extracted_formulas_json VARCHAR, 
    formulas_count INTEGER NOT NULL, 
    extracted_answer VARCHAR, 
    answer_confidence FLOAT, 
    solution_steps_json VARCHAR, 
    ocr_confidence FLOAT, 
    solution_structure_confidence FLOAT, 
    formula_confidence FLOAT, 
    answer_match_confidence FLOAT, 
    total_confidence FLOAT, 
    check_level VARCHAR NOT NULL, 
    suggested_grade FLOAT, 
    auto_feedback VARCHAR, 
    processing_time_ms INTEGER, 
    recognized_at TIMESTAMP WITHOUT TIME ZONE NOT NULL, 
    PRIMARY KEY (id), 
    FOREIGN KEY(image_id) REFERENCES assessmentimage (id)
);

CREATE TABLE assessmentresult (
    id SERIAL NOT NULL, 
    solution_id INTEGER NOT NULL, 
    teacher_id INTEGER, 
    teacher_verdict VARCHAR, 
    teacher_score FLOAT, 
    teacher_feedback VARCHAR, 
    corrected_text VARCHAR, 
    corrected_formulas_json VARCHAR, 
    corrected_answer VARCHAR, 
    system_score FLOAT, 
    system_feedback VARCHAR, 
    used_for_training BOOLEAN NOT NULL, 
    training_priority INTEGER NOT NULL, 
    verified_at TIMESTAMP WITHOUT TIME ZONE, 
    created_at TIMESTAMP WITHOUT TIME ZONE NOT NULL, 
    PRIMARY KEY (id), 
    FOREIGN KEY(solution_id) REFERENCES recognizedsolution (id), 
    FOREIGN KEY(teacher_id) REFERENCES "user" (id)
);

CREATE TABLE trainingsample (
    id SERIAL NOT NULL, 
    image_path VARCHAR NOT NULL, 
    original_text VARCHAR NOT NULL, 
    corrected_text VARCHAR NOT NULL, 
    subject VARCHAR, 
    handwriting_style VARCHAR, 
    image_quality VARCHAR, 
    used_in_training BOOLEAN NOT NULL, 
    training_iteration INTEGER, 
    model_version VARCHAR, 
    collected_at TIMESTAMP WITHOUT TIME ZONE NOT NULL, 
    used_at TIMESTAMP WITHOUT TIME ZONE, 
    PRIMARY KEY (id)
);

CREATE TABLE systemmetrics (
    id SERIAL NOT NULL, 
    date TIMESTAMP WITHOUT TIME ZONE NOT NULL, 
    total_processed INTEGER NOT NULL, 
    level_1_count INTEGER NOT NULL, 
    level_2_count INTEGER NOT NULL, 
    level_3_count INTEGER NOT NULL, 
    avg_ocr_confidence FLOAT, 
    avg_processing_time_ms FLOAT, 
    error_count INTEGER NOT NULL, 
    most_common_error VARCHAR, 
    PRIMARY KEY (id)
);

CREATE INDEX ix_assessmentimage_assignment_id ON assessmentimage (assignment_id);

CREATE INDEX ix_assessmentimage_class_id ON assessmentimage (class_id);

CREATE INDEX ix_assessmentimage_student_id ON assessmentimage (student_id);

CREATE INDEX ix_assessmentimage_status ON assessmentimage (status);

CREATE INDEX ix_recognizedsolution_image_id ON recognizedsolution (image_id);

CREATE INDEX ix_recognizedsolution_check_level ON recognizedsolution (check_level);

CREATE INDEX ix_recognizedsolution_total_confidence ON recognizedsolution (total_confidence);

CREATE INDEX ix_assessmentresult_solution_id ON assessmentresult (solution_id);

CREATE INDEX ix_assessmentresult_teacher_id ON assessmentresult (teacher_id);

UPDATE alembic_version SET version_num='add_assessit_models' WHERE alembic_version.version_num = 'add_name_surname_to_user';

COMMIT;

