use std::{slice, sync::Arc, vec};
// 新增在 tensor.rs 顶部
use half::f16;
#[derive(Debug, Clone, Copy)]
pub enum DType {
    F16,
    F32,
}

pub struct Tensor<T> {
    data: Arc<Box<[T]>>,
    shape: Vec<usize>,
    offset: usize,
    length: usize,
}

impl<T: Copy + Clone + Default + Into<f32>> Tensor<T> {
    pub fn to_dtype(&self, dtype: DType) -> Tensor<f32> {
        match dtype {
            DType::F16 => {
                let converted: Vec<f16> = self
                    .data()
                    .iter()
                    .map(|x| f16::from_f32((*x).into()))
                    .collect();
                let f32_data: Vec<f32> = converted.iter().map(|x| x.to_f32()).collect();
                Tensor::new(f32_data, &self.shape)
            }
            DType::F32 => Tensor::new(
                self.data().iter().map(|x| (*x).into()).collect(),
                &self.shape,
            ),
        }
    }
}

impl<T: Copy + Clone + Default> Tensor<T> {
    pub fn new(data: Vec<T>, shape: &[usize]) -> Self {
        let length = data.len();
        Tensor {
            data: Arc::new(data.into_boxed_slice()),
            shape: shape.to_owned(),
            offset: 0,
            length,
        }
    }

    pub fn default(shape: &[usize]) -> Self {
        let length = shape.iter().product();
        let data = vec![T::default(); length];
        Self::new(data, shape)
    }

    pub fn data(&self) -> &[T] {
        //? 等价于self.data[self.offset:self.offset + self.length]
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
