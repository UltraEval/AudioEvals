In the Quick Start section, you can initiate an evaluation task within just a few seconds. However, delving deeper into the process, here's a more detailed breakdown of what's involved when you actually launch an evaluation task:

![assets/img_1.png](../assets/img_1.png)

When initiating an evaluation task, you will need to provide the following components:
- **Dataset**: This is the data set against which the model's performance will be evaluated. It should be representative of the real-world scenarios the model is expected to handle.
- **Prompt**: The prompt serves as a converter, translating data directives into model inputs.
 We have standardized the prompt format to facilitate prompt management, ensuring that users need not be concerned with the intricacies of the underlying model's input style.
- **Model**: The model to be evaluated. 
- **Post-Process**: This step involves any necessary processing or transformations applied to the model's output before evaluation. This could include formatting, filtering, or normalization.
- **Evaluator**: The evaluator serves as the mechanism for scoring each piece of data, simplifying the identification of problematic cases.
- **Aggregation**:Distinct from the Evaluator in scope, this component operates at the dataset level.

