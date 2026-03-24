#!/usr/bin/env nextflow

nextflow.enable.dsl=2

// Pipeline Parametreleri
params.input_dir = "${projectDir}/data/raw"
params.output_dir = "${projectDir}/data/processed"
params.script = "${projectDir}/src/pipeline/preprocess.py"

// Process: Feature Extractor scriptini her CSV dosyası için paralel koştur.
process PREPROCESS_DATA {
    // Conda ortamını kullan
    conda "${projectDir}/environment.yml"
    
    // Çıktıları kopyalama moduyla (symlink yerine) output dizinine gönder
    publishDir params.output_dir, mode: 'copy'
    
    input:
    path raw_csv
    
    output:
    path "processed_${raw_csv}"
    
    script:
    """
    python ${params.script} --input ${raw_csv} --output processed_${raw_csv}
    """
}

workflow {
    // raw data altındaki tüm csv dosyalarını bir kanala (channel) doldurur
    input_files_ch = Channel.fromPath("${params.input_dir}/*.csv")
    
    // Processi kanal üzerinden koştur
    PREPROCESS_DATA(input_files_ch)
}
