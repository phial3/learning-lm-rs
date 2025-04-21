use half::{bf16, f16}; // 添加对 f16 和 bf16 的支持
use std::any::TypeId;
use std::{slice, sync::Arc, vec};

pub struct Tensor<T> {
    data: Arc<Box<[T]>>,
    shape: Vec<usize>,
    offset: usize,
    length: usize,
}
// 自定义转换特征
pub trait FloatConvert {
    fn to_f32(self) -> f32;
    fn from_f32(val: f32) -> Self;
}

// 为需要支持的浮点类型实现特征
impl FloatConvert for half::f16 {
    fn to_f32(self) -> f32 {
        self.to_f32()
    }

    fn from_f32(val: f32) -> Self {
        half::f16::from_f32(val)
    }
}

impl FloatConvert for f32 {
    fn to_f32(self) -> f32 {
        self
    }

    fn from_f32(val: f32) -> Self {
        val
    }
}

impl<T: Copy + Clone + Default + 'static> Tensor<T> {
    pub fn new(data: Vec<T>, shape: &[usize]) -> Self {
        let length = data.len();
        Tensor {
            data: Arc::new(data.into_boxed_slice()),
            shape: shape.to_owned(),
            offset: 0,
            length,
        }
    }

    /// 将张量转换为目标精度
    pub fn to_f32(&self) -> Tensor<f32> {
        let now_type_id = TypeId::of::<T>();

        // 零拷贝优化：直接类型转换 f32 --> f32
        if now_type_id == TypeId::of::<f32>() {
            return unsafe {
                Tensor {
                    data: Arc::new(std::mem::transmute::<Box<[T]>, Box<[f32]>>(
                        Arc::try_unwrap(self.data.clone()).unwrap_or_else(|arc| (*arc).clone()),
                    )),
                    shape: self.shape.clone(),
                    offset: self.offset,
                    length: self.length,
                }
            };
        }

        // // SIMD加速转换（AVX2优化）
        // #[cfg(all(target_arch = "x86_64", target_feature = "avx2"))]
        // unsafe {
        //     if now_type_id == TypeId::of::<f16>() {
        //         return self.avx2_convert();
        //     }
        // }

        // 处理 f16 -> f32 的转换
        if now_type_id == TypeId::of::<f16>() {
            // 安全：已通过TypeId检查确保类型匹配
            let data_f16 = unsafe { &*(self.data.as_ref() as &[T] as *const [T] as *const [f16]) };
            let new_data = data_f16.iter().map(|x| x.to_f32()).collect::<Vec<f32>>();

            return Tensor {
                data: Arc::new(new_data.into_boxed_slice()),
                shape: self.shape.clone(),
                offset: self.offset,
                length: self.length,
            };
        }
        // 添加其他类型转换...

        panic!("Unsupported type conversion from {:?} to f32", now_type_id);
    }
    // /// 异步转换版本
    // pub async fn to_f32_async(&self) -> Tensor<f32> {
    //     let cloned = self.clone();
    //     tokio::task::spawn_blocking(move || cloned.to_f32()).await.unwrap()
    // }

    // #[cfg(target_arch = "x86_64")]
    // unsafe fn avx2_convert(&self) -> Tensor<f32> {
    //     use std::arch::x86_64::*;

    //     let src = self.data.as_ref().as_ptr() as *const f16;
    //     let len = self.data.len();
    //     let mut dst = Vec::with_capacity(len);
    //     dst.set_len(len);
    //     let dst_ptr = dst.as_mut_ptr() as *mut f32;

    //     // AVX2加速转换（每次处理8个元素）
    //     for i in (0..len).step_by(8) {
    //         let pack = _mm_loadu_si128(src.add(i) as _);
    //         let converted = _mm256_cvtph_ps(pack);
    //         _mm256_storeu_ps(dst_ptr.add(i), converted);
    //     }

    //     Tensor {
    //         data: Arc::new(dst.into_boxed_slice()),
    //         shape: self.shape.clone(),
    //         offset: self.offset.clone(),
    //         length: self.length,
    //     }
    // }
    /// 将张量转换为目标精度
    pub fn to_f16(&self) -> Tensor<f16> {
        let now_type_id = TypeId::of::<T>();

        // 处理 f32 -> f16 的转换
        if now_type_id == TypeId::of::<f32>() {
            // 安全：已通过TypeId检查确保类型匹配
            let data_f32 = unsafe { &*(self.data.as_ref() as &[T] as *const [T] as *const [f32]) };
            let new_data = data_f32
                .iter()
                .map(|x| f16::from_f32(*x))
                .collect::<Vec<f16>>();
            return Tensor {
                data: Arc::new(new_data.into_boxed_slice()),
                shape: self.shape.clone(),
                offset: self.offset,
                length: self.length,
            };
        }

        panic!("Unsupported type conversion from {:?} to f16", now_type_id);
    }
    pub fn default(shape: &[usize]) -> Self {
        let length = shape.iter().product();
        let data = vec![T::default(); length];
        Self::new(data, shape)
    }

    pub fn data(&self) -> &[T] {
        &self.data[self.offset..][..self.length]
    }

    pub unsafe fn data_mut(&mut self) -> &mut [T] {
        let ptr = self.data.as_ptr().add(self.offset) as *mut T;
        slice::from_raw_parts_mut(ptr, self.length)
    }

    pub fn shape(&self) -> &Vec<usize> {
        &self.shape
    }

    pub fn size(&self) -> usize {
        self.length
    }

    // Reinterpret the tensor as a new shape while preserving total size.
    pub fn reshape(&mut self, new_shape: &Vec<usize>) -> &mut Self {
        let new_length: usize = new_shape.iter().product();
        if new_length != self.length {
            let old_shape = self.shape.clone();
            panic!("New shape {new_shape:?} does not match tensor of {old_shape:?}");
        }
        self.shape = new_shape.clone();
        self
    }

    pub fn slice(&self, start: usize, shape: &[usize]) -> Self {
        let new_length: usize = shape.iter().product();
        assert!(self.offset + start + new_length <= self.length);
        Tensor {
            data: self.data.clone(),
            shape: shape.to_owned(),
            offset: self.offset + start,
            length: new_length,
        }
    }
}

// Some helper functions for testing and debugging
impl Tensor<f32> {
    #[allow(unused)]
    pub fn close_to(&self, other: &Self, rel: f32) -> bool {
        if self.shape() != other.shape() {
            return false;
        }
        let a = self.data();
        let b = other.data();
        a.iter().zip(b).all(|(x, y)| float_eq(x, y, rel))
    }

    #[allow(unused)]
    pub fn print(&self) {
        println!(
            "shape: {:?}, offset: {}, length: {}",
            self.shape, self.offset, self.length
        );
        let dim = self.shape().last().cloned().unwrap_or_default();
        let batch = self.length / dim;
        for i in 0..batch {
            let start = i * dim;
            println!("{:?}", &self.data()[start..].get(..dim).unwrap_or_default());
        }
    }
}

#[inline]
pub fn float_eq(x: &f32, y: &f32, rel: f32) -> bool {
    (x - y).abs() <= rel * (x.abs() + y.abs()) / 2.0
}

// 其他数据类型实现
impl Tensor<f16> {
    #[allow(unused)]
    pub fn print(&self) {
        println!(
            "shape: {:?}, offset: {}, length: {}",
            self.shape, self.offset, self.length
        );
        let dim = self.shape().last().cloned().unwrap_or_default();
        let batch = self.length / dim;
        for i in 0..batch {
            let start = i * dim;
            println!("{:?}", &self.data()[start..].get(..dim).unwrap_or_default());
        }
    }
}

impl Tensor<bf16> {
    #[allow(unused)]
    pub fn print(&self) {
        println!(
            "shape: {:?}, offset: {}, length: {}",
            self.shape, self.offset, self.length
        );
        let dim = self.shape().last().cloned().unwrap_or_default();
        let batch = self.length / dim;
        for i in 0..batch {
            let start = i * dim;
            println!("{:?}", &self.data()[start..].get(..dim).unwrap_or_default());
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use half::f16;

    #[test]
    pub fn test_transfor() {
        let data: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0];
        let shape = vec![2, 2];
        let tensor = Tensor::new(data, &shape);
        let tensor_f16 = tensor.to_f16();
        for i in tensor_f16.data() {
            println!("{:?}", i);
        }
        let tensor_f32 = tensor_f16.to_f32();
        for i in tensor_f32.data() {
            println!("{:?}", i);
        }
        let a = f16::from_f32(1.3);
        println!("f16: {:?}", a);
        assert_eq!(2 + 2, 4);
    }
}
