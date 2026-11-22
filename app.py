import pickle
import gradio as gr  # type: ignore[import-not-found]
import spaces
# Load the model and scaler
regmodel = pickle.load(open('regmodel.pkl', 'rb'))
scalar = pickle.load(open('scaling.pkl', 'rb'))

@spaces.GPU
def predict_price(MedInc, HouseAge, AveRooms, AveBedrms, Population, AveOccup, Latitude, Longitude):
    data = [[MedInc, HouseAge, AveRooms, AveBedrms, Population, AveOccup, Latitude, Longitude]]
    final_input = scalar.transform(data)
    output = regmodel.predict(final_input)
    return f"The House price prediction is {output[0]:.4f}"

with gr.Blocks(title="California House Price Prediction") as demo:
    gr.Markdown("# California House Price Prediction (CI/CD Live Test)")
    with gr.Row():
        with gr.Column():
            MedInc = gr.Number(label="MedInc", value=8.3252)
            HouseAge = gr.Number(label="HouseAge", value=41.0)
            AveRooms = gr.Number(label="AveRooms", value=6.9841)
            AveBedrms = gr.Number(label="AveBedrms", value=1.0238)
        with gr.Column():
            Population = gr.Number(label="Population", value=322.0)
            AveOccup = gr.Number(label="AveOccup", value=2.5556)
            Latitude = gr.Number(label="Latitude", value=37.88)
            Longitude = gr.Number(label="Longitude", value=-122.23)

    predict_btn = gr.Button("Predict", variant="primary")
    output_text = gr.Textbox(label="Prediction Result")

    predict_btn.click(
        fn=predict_price,
        inputs=[MedInc, HouseAge, AveRooms, AveBedrms, Population, AveOccup, Latitude, Longitude],
        outputs=output_text
    )

if __name__ == "__main__":
    demo.launch()