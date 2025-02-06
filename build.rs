fn main() {
    println!("cargo:rerun-if-changed=build.rs");
    #[cfg(feature = "cuda")]
    {
        let builder = bindgen_cuda::Builder::default().kernel_paths_glob("cuda-kernels/**/*.cu");
        println!("cargo:info={builder:?}");
        let bindings = builder.build_ptx().unwrap();
        bindings
            .write("src/operators_cuda/cuda_kernels.rs")
            .unwrap();
    }
}
