pub struct LinearLayer {
    weights: Vec<Vec<f32>>,
    biases: Vec<f32>,
}

impl LinearLayer {
    pub fn new(weights: Vec<Vec<f32>>, biases: Vec<f32>) -> Self {
        LinearLayer { weights, biases }
    }
}