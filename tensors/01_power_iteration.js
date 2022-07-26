/**
 * Power iteration example using tensorflow.js
 * Author: Juan Manuel Pérez
 */

const tf = require('@tensorflow/tfjs');


// Power iteration for a matrix

// First, let's create a reflection matrix
let D = tf.tensor([
    [35.0, 0.0, 0.0],
    [0.0, 1.0, 0.0],
    [0.0, 0.0, 0.5]
]);

// 3x3 identity matrix
let I = tf.eye(3);
// Orthogonal matrix
let v = tf.tensor([1, 1, 1])
let squaredNorm = 3;
let B = I.add(tf.outerProduct(v, v).div(squaredNorm).mul(-2));

// This is the matrix we want to take the eigenvalues of
let X = B.matMul(D).matMul(B.transpose())

const maxIter = 100;
// current vector
let b = tf.tensor([1, 0.5, 0.25]).reshape([-1, 1]);

for (let count=0; count < maxIter; ++count) {
    // Power iteration
    b = X.matMul(b);
    b = b.div(b.norm());
}

let num = tf.dot(b.reshape([-1]), X.matMul(b).reshape([-1])).dataSync()[0];
let den = tf.dot(b.reshape([-1]), b.reshape([-1])).dataSync()[0];
let eigenvalue = num / den;
console.log(`Eigenvalue: ${eigenvalue}`);
