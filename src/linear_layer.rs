pub struct LinearLayer {
    weights: Vec<Vec<f32>>,
    biases: Vec<f32>,
}

impl LinearLayer {
    pub fn new(weights: Vec<Vec<f32>>, biases: Vec<f32>) -> Self {
        LinearLayer { weights, biases }
    }

    // Performs the forward pass of the linear layer
    // input: a vector of input values
    // output: a vector of output values after applying the linear transformation
    // output[i] = sum(weights[i][j] * input[j]) + biases[i]
    pub fn forward(&self, inputs: &Vec<f32>) -> Vec<f32> {
        if inputs.len() != self.weights[0].len() {
            panic!("Input size does not match weight dimensions");
        }
        
        self
            .weights
            .iter()
            .zip(&self.biases)
            .map(|(weight_row, bias)| {
                weight_row
                    .iter()
                    .zip(inputs)
                    .map(|(w, i)| w * i)
                    .sum::<f32>()
                    + bias
            })
            .collect::<Vec<f32>>()
    }
}
