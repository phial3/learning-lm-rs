use half::f16;
use std::{slice, sync::Arc, vec};

pub struct Tensor<T> {
    // Arc类似c++中的shared_ptr
    // Box类似c++中的unique_ptr, 表示堆上的数据
    data: Arc<Box<[T]>>,
    shape: Vec<usize>,
    offset: usize, // offset为了实现slice服务
    length: usize, // number of elements
}

impl<T: Copy + Clone + Default> Tensor<T> {
    pub fn new(data: Vec<T>, shape: &[usize]) -> Self {
        let length = data.len();
        Tensor {
            // into_boxed_slice()将Vec<T>转换为Box<[T]>
            // try_into()将Box<[T]>转换为Box<[T; N]>
            // unwrap()Box<T>转换为T
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

impl Tensor<f16> {
    #[allow(unstable_features)]
    pub fn to_f32(&self) -> Tensor<f32> {
        let f32_data: Vec<f32> = self.data().iter().map(|x| x.to_f32()).collect();
        Tensor::new(f32_data, &self.shape)
    }
}

impl Tensor<f32> {
    #[allow(unstable_features)]
    pub fn to_f16(&self) -> Tensor<f16> {
        let f16_data: Vec<f16> = self.data().iter().map(|x| f16::from_f32(*x)).collect();
        Tensor::new(f16_data, &self.shape)
    }
}

#[inline]
pub fn float_eq(x: &f32, y: &f32, rel: f32) -> bool {
    (x - y).abs() <= rel * (x.abs() + y.abs()) / 2.0
}
