[video coming soon]

Welcome! This guide will walk you through the concept of gradients. We'll start with the familiar idea of a derivative and build up to understanding how gradients make neural networks learn.

Prerequisites: Check out previous 3 lessons: Functions, Derivatives & Vectors



Step 1: From Line Slope (Derivative) To Surface Slope (Gradient)

Let's start with what you know. For a simple function like f(x) = x², the derivative f'(x) = 2x gives you the slope of the curve at any point x. So for x=3, derivative is 2*3=6. That means as you increase x but a tiny bit, f(x) = x² will increase by 6.

At x=4, derivative is 2*4=8, so at that point f(x) = x² is increasing by 8x.





Notice that I say "if you increase x by a bit, f(x) = x² will increase by 6" and I don't say "if you increase x by 1", because increasing x by 1 (from 3 to 4 in this case) is a lot and by that point derivative (rate of change) will go from 6 to 8.

On this image you can see that the red slope at x=3 is smaller than thes green slope at x=4.

In this case, if you increase x=3 by 1, derivative will go from 6 to 8. So that's why we say "if you increase x=3 by a tiny bit, f(x) = x² will increase by 6".

But what if our function has multiple inputs, like f(x, y) = x² + y²?





This function doesn't describe a line; it describes a 3D surface, like a bowl landscape. If you're standing at any point (x, y) on this surface, what is "the" slope?

There isn't just one. There's a slope if you take a step in the x-direction, a different slope if you step in the y-direction, and another for every other direction in between.

To handle this, we use partial derivatives.





Partial Derivative with respect to x (∂f/∂x): This is the slope if you only move in the x-direction. You treat y as a constant. For f(x, y) = x² + y², the partial derivative ∂f/∂x = 2x - remember the rule for a constant that stands alone, constants become 0 in the derivative, and since we treat y as a constant, + y² will ecome + 0.



Partial Derivative with respect to y (∂f/∂y): This is the slope if you only move in the y-direction. You treat x as a constant. For f(x, y) = x² + y², the partial derivative ∂f/∂y = 2y.

Now we have two slopes, one for each axis. The gradient is simply a way to package all these partial derivatives together.

Definition: The gradient is a vector that contains all the partial derivatives of a function. It's denoted by ∇f (pronounced "nabla f" or "del f").

For our function f(x, y), the gradient is:

∇f = [ ∂f/∂x, ∂f/∂y ] = [ 2x, 2y ]



Step 2: What the Gradient Vector Tells Us

So, the gradient is a vector (think of it as an arrow). What do the direction and length of this arrow mean?

This is the most important intuition to grasp.

1. The Direction of the Gradient

The gradient vector at any point (x, y) points in the direction of the steepest possible ascent.

Imagine you're standing on a mountainside. If you look around, there are many ways to take a step. One direction leads straight uphill, another leads straight downhill, and others traverse the mountain at a constant elevation. The gradient is an arrow painted on the ground at your feet that points directly up the steepest path from where you are.

2. The Magnitude (Length) of the Gradient

The length of the gradient vector tells you how steep that steepest path is.





A long gradient vector means the slope is very steep. A small step will result in a large change in elevation.



A short gradient vector means the slope is gentle. The terrain is nearly flat.



A zero-length gradient vector (i.e., [0, 0]) means you are at a flat spot—either a peak, a valley bottom, or a flat plateau.



Step 3: A Concrete Example

Let's go back to our bowl function, f(x, y) = x² + y², and its gradient, ∇f = [2x, 2y]. The minimum of this function is clearly at (0, 0).

Let's calculate the gradient at a specific point, say (3, 1).





∇f(3, 1) = [ 2  3, 2  1 ] = [6, 2]

This vector [6, 2] is an arrow that points "6 units in the x-direction and 2 units in the y-direction." This is an arrow pointing up and to the right, away from the minimum at (0, 0). This makes perfect sense! From the point (3, 1), the steepest way up the bowl is away from the bottom.

What about the point (-2, -2)?





∇f(-2, -2) = [ 2  -2, 2  -2 ] = [-4, -4]

This vector points down and to the left, again, away from the bottom of the bowl at (0, 0).



Step 4: Visualizing the Gradient Field

Let's visualize this. The image below shows a contour plot of our function f(x, y) = x² + y². Think of this as a topographic map. The lines connect points of equal "elevation." The arrows represent the gradient vectors at various points.

Notice two crucial properties in the visualization:





Direction: The arrows always point from a lower contour line to a higher one (from blue to yellow). They show the path of steepest ascent.



Orthogonality: The gradient vectors are always perpendicular to the contour lines. To go straight uphill, you must walk at a right angle to the path of "no elevation change."





When you run this, you will see a visual representation of everything we've discussed.



Step 5: The "Why": Gradients and Machine Learning

This is where gradients become incredibly powerful. In machine learning, we define a loss function (or cost function). This function measures how "wrong" our model's predictions are. The inputs to this function are the model's parameters (its weights and biases), and the output is a single number representing the total error.

Our goal is to find the set of parameters that minimizes the error.

This is the exact same problem as finding the lowest point in a valley!

The algorithm used to do this is called Gradient Descent. Here’s how it works:





Start Somewhere: Initialize the model's parameters to random values. (This is like dropping a hiker at a random spot on the mountain).



Find the Way Down: Calculate the gradient of the loss function at your current location. The gradient points straight uphill.



Take a Step Downhill: To go downhill, simply move in the direction of the negative gradient. We update our parameters by taking a small step in that opposite direction.



Repeat: Go back to step 2. Keep calculating the gradient and taking small steps downhill until you reach the bottom of the valley, where the gradient is zero.

This is the core mechanic of how neural networks "learn." They are constantly calculating the gradient of their error and adjusting their internal parameters to move in the direction that reduces that error.

Key Takeaways





A gradient is a vector of partial derivatives that generalizes the concept of slope to functions with multiple inputs.



Direction: The gradient vector points in the direction of the steepest ascent.



Magnitude: Its length represents how steep that ascent is.



Optimization: The negative gradient points in the direction of steepest descent, which is the key to finding the minimum of a function using Gradient Descent.