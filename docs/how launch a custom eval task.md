In the Quick Start section, you can initiate an evaluation task within just a few seconds. However, delving deeper into the process, here's a more detailed breakdown of what's involved when you actually launch an evaluation task:

![assets/img_1.png](../assets/img_1.png)

When initiating an evaluation task, you will need to provide the following components:
- **Dataset**: This is the data set against which the model's performance will be evaluated. It should be representative of the real-world scenarios the model is expected to handle. Build-in Config Path:  [dataset](../registry/dataset)
- **Prompt**: The prompt serves as a converter, translating data directives into model inputs. 
 We have standardized the prompt format to facilitate prompt management, ensuring that users need not be concerned with the intricacies of the underlying model's input style. Build-in Config Path: [prompt](../registry/prompt)
- **Model**: The model to be evaluated. Build-in Config Path: [model](../registry/model)
- **Post-Process**: This step involves any necessary processing or transformations applied to the model's output before evaluation. This could include formatting, filtering, or normalization. Build-in Config Path: [post-process](../registry/process)
- **Evaluator**: The evaluator serves as the mechanism for scoring each piece of data, simplifying the identification of problematic cases. Build-in Config Path: [evaluator](../registry/evaluator)
- **Aggregation**:Distinct from the Evaluator in scope, this component operates at the dataset level. It aggregates the results of the evaluation across all data points, providing a comprehensive view of the model's performance. Build-in Config Path: [agg](../registry/agg)
