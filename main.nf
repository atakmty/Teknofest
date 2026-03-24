#!/usr/bin/env nextflow

nextflow.enable.dsl=2

// Pipeline Parametreleri
params.input_dir = "${projectDir}/data/raw"
params.output_dir = "${projectDir}/data/processed"
params.model_dir = "${projectDir}/models"
params.script = "${projectDir}/src/pipeline/preprocess.py"

// Process 1: TRAIN
process PROCESS_TRAIN {
    conda "${projectDir}/environment.yml"
    publishDir params.output_dir, mode: 'copy', pattern: '*.csv'
    publishDir params.model_dir, mode: 'copy', pattern: '*.pkl'
    
    input:
    tuple val(dataset_name), path(raw_csv)
    
    output:
    tuple val(dataset_name), path("processed_${raw_csv}"), path("${dataset_name}_extractor.pkl"), emit: train_results
    
    script:
    """
    python ${params.script} --input ${raw_csv} --output processed_${raw_csv} --mode train --model_path ${dataset_name}_extractor.pkl
    """
}

// Process 2: TEST
process PROCESS_TEST {
    conda "${projectDir}/environment.yml"
    publishDir params.output_dir, mode: 'copy', pattern: '*.csv'
    
    input:
    tuple val(dataset_name), path(test_csv), path(extractor_pkl)
    
    output:
    path "processed_${test_csv}"
    
    script:
    """
    python ${params.script} --input ${test_csv} --output processed_${test_csv} --mode test --model_path ${extractor_pkl}
    """
}

workflow {
    Channel
        .fromPath("${params.input_dir}/*_train.csv")
        .map { file -> tuple(file.simpleName.replaceAll('_train$', ''), file) }
        .set { train_files_ch }

    Channel
        .fromPath("${params.input_dir}/*_test.csv")
        .map { file -> tuple(file.simpleName.replaceAll('_test$', ''), file) }
        .set { test_files_ch }

    PROCESS_TRAIN(train_files_ch)

    model_ch = PROCESS_TRAIN.out.train_results.map { it -> tuple(it[0], it[2]) }
    test_joined_ch = test_files_ch.join(model_ch)

    PROCESS_TEST(test_joined_ch)
}
