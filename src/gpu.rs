use crate::tensor::Tensor;
use cust::launch;
use cust::memory::DeviceBuffer;
use cust::memory::DeviceCopy;
use cust::prelude::*;
use cust::sys;
use once_cell::sync::Lazy;
use std::error::Error;
use std::fmt;
use std::slice;

const KERNELS_PTX: &str = include_str!("kernels.ptx");

static CUDA_CONTEXT: Lazy<Context> =
    Lazy::new(|| cust::quick_init().expect("CUDA quick_init failed"));

static CUDA_MODULE: Lazy<Module> = Lazy::new(|| {
    let _ctx = &*CUDA_CONTEXT;
    Module::from_ptx(KERNELS_PTX, &[]).expect("Module load failed")
});

/// GPU 错误类型
#[derive(Debug)]
pub struct GpuError(String);

impl fmt::Display for GpuError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "GpuError: {}", self.0)
    }
}

impl Error for GpuError {}

/// gather_kernel
pub fn gather_kernel<U: Copy + Default + DeviceCopy>(
    y: &mut Tensor<f32>,
    indices: &Tensor<u32>,
    table: &Tensor<U>,
) -> Result<(), GpuError> {
    unsafe {
        if sys::cuCtxSetCurrent((&*CUDA_CONTEXT).as_raw()) != sys::CUresult::CUDA_SUCCESS {
            return Err(GpuError("Unable to set current CUDA context".into()));
        }
    }
    let length = indices.size();
    let table_shape = table.shape();
    if table_shape.len() != 2 {
        return Err(GpuError("Table must be 2D".to_string()));
    }
    let dim = table_shape[1];
    if y.size() != length * dim {
        return Err(GpuError(
            "Output tensor size does not match indices and table".to_string(),
        ));
    }

    let module = load_module()?;
    let stream =
        Stream::new(StreamFlags::NON_BLOCKING, None).map_err(|e| GpuError(e.to_string()))?;

    let d_table = DeviceBuffer::from_slice(table.data()).map_err(|e| GpuError(e.to_string()))?;
    let d_indices =
        DeviceBuffer::from_slice(indices.data()).map_err(|e| GpuError(e.to_string()))?;
    let mut d_y = unsafe { DeviceBuffer::<f32>::uninitialized(y.size()) }
        .map_err(|e| GpuError(e.to_string()))?;
    let function = module
        .get_function("gather_kernel")
        .map_err(|e| GpuError(e.to_string()))?;

    let block_size = 256;
    let grid_size = ((length as u32) + block_size - 1) / block_size;

    unsafe {
        launch!(function<<<grid_size, block_size, 0, stream>>>(
            d_table.as_device_ptr(),
            d_indices.as_device_ptr(),
            d_y.as_device_ptr(),
            dim as u32,
            length as u32
        ))
        .map_err(|e| GpuError(e.to_string()))?;
    }
    stream.synchronize().map_err(|e| GpuError(e.to_string()))?;
    let y_slice = unsafe { slice::from_raw_parts_mut(y.data_mut().as_mut_ptr(), y.size()) };
    d_y.copy_to(y_slice).map_err(|e| GpuError(e.to_string()))?;
    Ok(())
}

/// rope_kernel
pub fn rope_kernel(y: &mut Tensor<f32>, start_pos: usize, theta: f32) -> Result<(), GpuError> {
    let shape = y.shape();
    if shape.len() != 3 {
        return Err(GpuError("Tensor shape must be 3D for rope".to_string()));
    }
    let seq_len = shape[0];
    let n_heads = shape[1];
    let d = shape[2];
    let total = seq_len * n_heads * d;

    let module = load_module()?;
    let stream =
        Stream::new(StreamFlags::NON_BLOCKING, None).map_err(|e| GpuError(e.to_string()))?;
    let mut d_y = DeviceBuffer::<f32>::from_slice(y.data()).map_err(|e| GpuError(e.to_string()))?;
    let function = module
        .get_function("rope_kernel")
        .map_err(|e| GpuError(e.to_string()))?;

    let block_size = 256;
    let grid_size = ((total as u32) + block_size - 1) / block_size;

    unsafe {
        launch!(function<<<grid_size, block_size, 0, stream>>>(
            d_y.as_device_ptr(),
            start_pos as u32,
            theta,
            seq_len as u32,
            n_heads as u32,
            d as u32
        ))
        .map_err(|e| GpuError(e.to_string()))?;
    }
    stream.synchronize().map_err(|e| GpuError(e.to_string()))?;
    d_y.copy_to(unsafe { slice::from_raw_parts_mut(y.data_mut().as_mut_ptr(), y.size()) })
        .map_err(|e| GpuError(e.to_string()))?;
    Ok(())
}

/// masked_softmax_kernel
pub fn masked_softmax_kernel(y: &mut Tensor<f32>) -> Result<(), GpuError> {
    let ndim = y.shape().len();
    if ndim < 2 {
        return Err(GpuError(
            "Tensor must be at least 2D for masked_softmax".to_string(),
        ));
    }
    let seq_len = y.shape()[ndim - 2];
    let total_seq_len = y.shape()[ndim - 1];
    let batch = y.size() / (seq_len * total_seq_len);

    let module = load_module()?;
    let stream =
        Stream::new(StreamFlags::NON_BLOCKING, None).map_err(|e| GpuError(e.to_string()))?;
    let mut d_y = DeviceBuffer::<f32>::from_slice(y.data()).map_err(|e| GpuError(e.to_string()))?;
    let function = module
        .get_function("masked_softmax_kernel")
        .map_err(|e| GpuError(e.to_string()))?;

    let block_size = seq_len as u32;
    let grid_size = batch as u32;

    unsafe {
        launch!(function<<<grid_size, block_size, 0, stream>>>(
            d_y.as_device_ptr(),
            batch as u32,
            seq_len as u32,
            total_seq_len as u32
        ))
        .map_err(|e| GpuError(e.to_string()))?;
    }
    stream.synchronize().map_err(|e| GpuError(e.to_string()))?;
    d_y.copy_to(unsafe { slice::from_raw_parts_mut(y.data_mut().as_mut_ptr(), y.size()) })
        .map_err(|e| GpuError(e.to_string()))?;
    Ok(())
}

// rms_norm_kernel
pub fn rms_norm_kernel(
    y: &mut Tensor<f32>,
    x: &Tensor<f32>,
    w: &Tensor<f32>,
    epsilon: f32,
) -> Result<(), GpuError> {
    let x_shape = x.shape();
    if x_shape.len() < 1 {
        return Err(GpuError("x must be at least 1D".to_string()));
    }
    let n = *x_shape.last().unwrap();
    if y.size() != x.size() || w.size() != n {
        return Err(GpuError("Size mismatch in rms_norm_kernel".to_string()));
    }
    let batch = x.size() / n;

    let module = load_module()?;
    let stream =
        Stream::new(StreamFlags::NON_BLOCKING, None).map_err(|e| GpuError(e.to_string()))?;

    let d_x = DeviceBuffer::<f32>::from_slice(x.data()).map_err(|e| GpuError(e.to_string()))?;
    let d_w = DeviceBuffer::<f32>::from_slice(w.data()).map_err(|e| GpuError(e.to_string()))?;
    let mut d_y = DeviceBuffer::<f32>::from_slice(y.data()).map_err(|e| GpuError(e.to_string()))?;
    let function = module
        .get_function("rms_norm_kernel")
        .map_err(|e| GpuError(e.to_string()))?;
    let block_size = n as u32;
    let grid_size = batch as u32;

    unsafe {
        launch!(function<<<grid_size, block_size, 0, stream>>>(
            d_y.as_device_ptr(),
            d_x.as_device_ptr(),
            d_w.as_device_ptr(),
            n as u32,
            epsilon,
            batch as u32
        ))
        .map_err(|e| GpuError(e.to_string()))?;
    }
    stream.synchronize().map_err(|e| GpuError(e.to_string()))?;
    d_y.copy_to(unsafe { slice::from_raw_parts_mut(y.data_mut().as_mut_ptr(), y.size()) })
        .map_err(|e| GpuError(e.to_string()))?;
    Ok(())
}

/// GPU 实现：swiglu_kernel
pub fn swiglu_kernel(y: &mut Tensor<f32>, x: &Tensor<f32>) -> Result<(), GpuError> {
    if y.size() != x.size() {
        return Err(GpuError("Size mismatch in swiglu_kernel".to_string()));
    }
    let len = y.size();

    let module = load_module()?;
    let stream =
        Stream::new(StreamFlags::NON_BLOCKING, None).map_err(|e| GpuError(e.to_string()))?;

    let d_x = DeviceBuffer::<f32>::from_slice(x.data()).map_err(|e| GpuError(e.to_string()))?;
    let mut d_y = DeviceBuffer::<f32>::from_slice(y.data()).map_err(|e| GpuError(e.to_string()))?;

    let function = module
        .get_function("swiglu_kernel")
        .map_err(|e| GpuError(e.to_string()))?;

    let block_size = 256;
    let grid_size = ((len as u32) + block_size - 1) / block_size;

    unsafe {
        launch!(function<<<grid_size, block_size, 0, stream>>>(
            d_y.as_device_ptr(),
            d_x.as_device_ptr(),
            len as u32
        ))
        .map_err(|e| GpuError(e.to_string()))?;
    }
    stream.synchronize().map_err(|e| GpuError(e.to_string()))?;
    d_y.copy_to(unsafe { slice::from_raw_parts_mut(y.data_mut().as_mut_ptr(), y.size()) })
        .map_err(|e| GpuError(e.to_string()))?;
    Ok(())
}

/// GPU 实现：matmul_transb_kernel
pub fn matmul_transb_kernel<U: Copy + Default + Sync + DeviceCopy>(
    c: &mut Tensor<f32>,
    beta: f32,
    a: &Tensor<f32>,
    b: &Tensor<U>,
    alpha: f32,
) -> Result<(), GpuError> {
    let a_shape = a.shape();
    if a_shape.len() != 2 {
        return Err(GpuError("A must be 2D".to_string()));
    }
    let m = a_shape[0];
    let k = a_shape[1];
    let b_shape = b.shape();
    if b_shape.len() != 2 || b_shape[1] != k {
        return Err(GpuError("B 的列数必须与 A 的列数相等".to_string()));
    }
    let n = b_shape[0];
    let c_shape = c.shape();
    if c_shape.len() != 2 || c_shape[0] != m || c_shape[1] != n {
        return Err(GpuError("C 的形状必须为 m×n".to_string()));
    }

    let module = load_module()?;
    let stream =
        Stream::new(StreamFlags::NON_BLOCKING, None).map_err(|e| GpuError(e.to_string()))?;

    let d_a = DeviceBuffer::<f32>::from_slice(a.data()).map_err(|e| GpuError(e.to_string()))?;
    // 注意：b 的数据类型为 U，通过 to_f32 后在 GPU 内部做计算（此处要求 U 的内存布局与 f32 相同）
    let d_b = DeviceBuffer::from_slice(b.data()).map_err(|e| GpuError(e.to_string()))?;
    let mut d_c = DeviceBuffer::<f32>::from_slice(c.data()).map_err(|e| GpuError(e.to_string()))?;

    let function = module
        .get_function("matmul_transb_kernel")
        .map_err(|e| GpuError(e.to_string()))?;

    let block_dim = (32, 32, 1);
    let grid_dim = (
        ((n as u32) + block_dim.0 - 1) / block_dim.0,
        ((m as u32) + block_dim.1 - 1) / block_dim.1,
        1,
    );

    unsafe {
        launch!(function<<<grid_dim, block_dim, 0, stream>>>(
            d_a.as_device_ptr(),
            d_b.as_device_ptr(),
            d_c.as_device_ptr(),
            m as u32,
            n as u32,
            k as u32,
            beta,
            alpha
        ))
        .map_err(|e| GpuError(e.to_string()))?;
    }
    stream.synchronize().map_err(|e| GpuError(e.to_string()))?;
    d_c.copy_to(unsafe { slice::from_raw_parts_mut(c.data_mut().as_mut_ptr(), c.size()) })
        .map_err(|e| GpuError(e.to_string()))?;
    Ok(())
}

// GPU实现flash attention
// 这里还可以实现一个flash attention对应的东西......

/// 加载 CUDA 模块（直接返回静态模块引用）
fn load_module() -> Result<&'static Module, GpuError> {
    Ok(&CUDA_MODULE)
}
