use half::{bf16, f16};
use num_traits::Float;
use std::{slice, sync::Arc, vec};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum DType {
    F16,
    BF16,
    F32,
}

#[derive(Clone)]
pub struct Tensor<T> {
    data: Arc<Box<[T]>>,
    shape: Vec<usize>,
    offset: usize,
    length: usize,
}

impl<T: Copy + Default> Tensor<T> {
    pub fn new(data: Vec<T>, shape: &Vec<usize>) -> Tensor<T> {
        let length = data.len();
        Tensor {
            data: Arc::new(data.into_boxed_slice().try_into().unwrap()),
            shape: shape.clone(),
            offset: 0,
            length: length,
        }
    }
    pub fn default(shape: &Vec<usize>) -> Self {
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

    pub fn slice(&self, start: usize, shape: &Vec<usize>) -> Self {
        let new_length: usize = shape.iter().product();
        assert!(self.offset + start + new_length <= self.length);
        Tensor {
            data: self.data.clone(),
            shape: shape.clone(),
            offset: self.offset + start,
            length: new_length,
        }
    }
}
pub trait PrecisionCast {
    const DTYPE: DType;
    fn cast_to_f32(&self) -> f32;
    fn cast_to_f16(&self) -> f16;
    fn cast_to_bf16(&self) -> bf16;
}

impl PrecisionCast for f32 {
    const DTYPE: DType = DType::F32;
    fn cast_to_f32(&self) -> f32 {
        *self
    }
    fn cast_to_f16(&self) -> f16 {
        f16::from_f32(*self)
    }
    fn cast_to_bf16(&self) -> bf16 {
        bf16::from_f32(*self)
    }
}

impl PrecisionCast for f16 {
    const DTYPE: DType = DType::F16;
    fn cast_to_f32(&self) -> f32 {
        self.to_f32()
    }
    fn cast_to_f16(&self) -> f16 {
        *self
    }
    fn cast_to_bf16(&self) -> bf16 {
        bf16::from_f32(self.to_f32())
    }
}

impl PrecisionCast for bf16 {
    const DTYPE: DType = DType::BF16;
    fn cast_to_f32(&self) -> f32 {
        self.to_f32()
    }
    fn cast_to_f16(&self) -> f16 {
        f16::from_f32(self.to_f32())
    }
    fn cast_to_bf16(&self) -> bf16 {
        *self
    }
}

impl<T: Copy + Default + PrecisionCast> Tensor<T> {
    pub fn dtype(&self) -> DType {
        T::DTYPE
    }

    pub fn cast_to<U: Copy + Default + PrecisionCast + Float>(&self) -> Tensor<U> {
        if T::DTYPE == U::DTYPE {
            // 如果源类型和目标类型相同，直接克隆
            unsafe {
                let data = std::slice::from_raw_parts(
                    self.data().as_ptr() as *const T as *const U,
                    self.length,
                )
                .to_vec();
                return Tensor::new(data, &self.shape);
            }
        }
        // 否则进行类型转换
        let data = match U::DTYPE {
            DType::F32 => self
                .data()
                .iter()
                .map(|x| U::from(x.cast_to_f32()).unwrap())
                .collect::<Vec<U>>(),
            DType::F16 => self
                .data()
                .iter()
                .map(|x| U::from(x.cast_to_f16()).unwrap())
                .collect::<Vec<U>>(),
            DType::BF16 => self
                .data()
                .iter()
                .map(|x| U::from(x.cast_to_bf16()).unwrap())
                .collect::<Vec<U>>(),
        };

        unsafe {
            let data = std::slice::from_raw_parts(data.as_ptr() as *const U, data.len()).to_vec();
            Tensor::new(data, &self.shape)
        }
    }

    pub fn to_f32(&self) -> Tensor<f32> {
        self.cast_to::<f32>()
    }

    pub fn to_f16(&self) -> Tensor<f16> {
        self.cast_to::<f16>()
    }

    pub fn to_bf16(&self) -> Tensor<bf16> {
        self.cast_to::<bf16>()
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

        return a.iter().zip(b).all(|(x, y)| float_eq(x, y, rel));
    }
    #[allow(unused)]
    pub fn print(&self) {
        println!(
            "shpae: {:?}, offset: {}, length: {}",
            self.shape, self.offset, self.length
        );
        let dim = self.shape()[self.shape().len() - 1];
        let batch = self.length / dim;
        for i in 0..batch {
            let start = i * dim;
            println!("{:?}", &self.data()[start..][..dim]);
        }
    }
}

#[inline]
pub fn float_eq(x: &f32, y: &f32, rel: f32) -> bool {
    (x - y).abs() <= rel * (x.abs() + y.abs()) / 2.0
}
