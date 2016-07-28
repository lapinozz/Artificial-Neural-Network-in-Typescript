class Neuron
{
    inputs: Neuron[];
    weights: number[];
    deltaWeights: number[];

    output: number;
    gradient: number;

    extraWeight: number;

    constructor(inputNeurons: Neuron[] = null, extraWeight: number = 0)
    {
        if(inputNeurons)
        {
            this.inputs = inputNeurons;
            this.weights = [];
            this.deltaWeights = [];

            this.extraWeight = extraWeight;

            for(let x = 0; x < this.inputs.length + extraWeight; x++)
            {
                this.weights.push(rand(-1, 1));
                this.deltaWeights.push(0);
            }
        }
    }

    computeOutput()
    {
        let sum: number = 0;
        for(let x = 0; x < this.inputs.length; x++)
            sum += this.inputs[x].output * this.weights[x];

        for(let x = this.inputs.length; x < this.weights.length; x++)
            sum += this.weights[x];

        this.output = Neuron.activate(sum);
    }

    computeOutputGradiant(targetValue: number)
    {
        let delta = targetValue - this.output;
        this.gradient = delta * Neuron.activateAlternative(this.output);
    }

    computeHiddenGradients(outputs: Neuron[])
    {
        let dow = 0.0;

        for(let x = 0; x < outputs.length; x++)
        {
            let index = outputs[x].inputs.indexOf(this);

            dow += outputs[x].weights[index] * outputs[x].gradient;
        }

        this.gradient = dow * Neuron.activateAlternative(this.output);
    }

    train(learningRate: number = 0.3, alpha: number = 0.5)
    {
        for(let x = 0; x < this.weights.length; x++)
        {
            let oldDeltaWeight = this.deltaWeights[x];
            let newDeltaWeight =
                // Individual input, magnified by the gradient and train rate:
                learningRate
                * (x < this.inputs.length ? this.inputs[x].output : 1)
                * this.gradient
                // Also add momentum = a fraction of the previous delta weight
                + alpha
                * oldDeltaWeight;

            this.deltaWeights[x] = newDeltaWeight;
            this.weights[x] += newDeltaWeight;
        }
    }

    static activate(sum: number)
    {
        return tanh(sum);
    }

    static activateAlternative(sum: number)
    {
        return 1.0 - tanh(sum) * tanh(sum);
    }

    static activateStrict(sum: number)
    {
        return (sum > 0) ? 1 : -1;
    }
}

class Network
{
    layers: Neuron[][];

    constructor(neuronPerLayer: number[], extraWeight: boolean)
    {
        this.layers = [];

        for(let neuronCount in neuronPerLayer)
        {
            let previousLayer = this.getOutputLayer();

            this.layers.push([]);

            for(let x = 0; x < neuronPerLayer[neuronCount]; x++)
                this.getOutputLayer().push(new Neuron(previousLayer, extraWeight ? 1 : 0));
        }
    }

    getInputLayer()
    {
        if(this.layers)
            return this.layers[0];
        else
            return [];
    }

    getOutputLayer()
    {
        if(this.layers)
            return this.layers[this.layers.length - 1];
        else
            return [];
    }

    getLayer(layerIndex: number)
    {
        return this.layers[layerIndex];
    }

    getLayerSize(layerIndex: number)
    {
        return this.getLayer(layerIndex).length;
    }
    getLayerCount()
    {
        return this.layers.length;
    }

    getNeuron(layerIndex: number, neuronIndex: number)
    {
        return this.layers[layerIndex][neuronIndex];
    }

    activate(inputs: number[])
    {
        for(let x in this.getLayer(0))
            this.getNeuron(0, +x).output = inputs[x];

        for(let x in this.layers)
        {
            if(x == '0')
                continue;

            for(let n of this.getLayer(+x))
                n.computeOutput();
        }

        let output: number[] = [];

        for(let n of this.getOutputLayer())
            output.push(n.output);

        return output;
    }

    train(targetOutput: number[])
    {
        for(let x in this.getOutputLayer())
            this.getOutputLayer()[x].computeOutputGradiant(targetOutput[x]);

        for(let x = this.getLayerCount() - 2; x > 0; x--)
        {
            for(let n of this.getLayer(x))
                n.computeHiddenGradients(this.getLayer(x + 1));
        }

        for(let x in this.layers)
        {
            if(x == '0')
                continue;

            for(let n of this.getLayer(+x))
                n.train();
        }
    }
}

class NetworkRenderer
{
    canvas: Canvas;
    network: Network;

    verticalSpacing = 140
    horizontalSpacing = 140;

    neuronRadius = 50;

    activeColor = new RGB(0, 128, 0);
    inactiveColor = new RGB(128, 0, 0);

    textColor = "white";

    pos: Vec2;

    constructor(network: Network, canvas: Canvas, pos: Vec2)
    {
        this.network = network;
        this.canvas = canvas;
        this.pos = pos;
    }

    getNeuronPosition(layerIndex: number, layerSize: number, neuronIndex: number)
    {
        let x = +layerIndex * this.horizontalSpacing;
        x += this.neuronRadius + this.neuronRadius * 0.1;

        let y = neuronIndex * this.verticalSpacing;
        y -= ((layerSize - 1) * this.verticalSpacing) / 2;

        return new Vec2(this.pos.x + x, this.pos.y + y);
    }

    getNeuronColor(n: Neuron)
    {
        return lerpRGB((n.output - 1.0) / -2.0, this.activeColor, this.inactiveColor);
    }

    draw()
    {
        for(let layerIndex in this.network.layers)
        {
            let previousLayer = this.network.getLayer(+layerIndex - 1);
            let layer = this.network.getLayer(+layerIndex);

            for(let n in layer)
            {
                let pos1 = this.getNeuronPosition(+layerIndex, layer.length, +n);

                if(!layer[n].inputs)
                    continue;

                for(let i in layer[n].inputs)
                {
                    let pos2 = this.getNeuronPosition(+layerIndex - 1, previousLayer.length, +i);

                    this.drawNeuronConnection(layer[n], layer[n].inputs[i], pos1, pos2);
                }
            }
        }

        for(let layerIndex in this.network.layers)
        {
            let layer = this.network.getLayer(+layerIndex);

            for(let n in layer)
            {
                let pos = this.getNeuronPosition(+layerIndex, layer.length, +n);

                this.drawNeuron(layer[n], pos);
            }
        }
    }

    drawNeuron(n: Neuron, pos: Vec2)
    {
        let color = this.getNeuronColor(n);

        this.canvas.drawCircle(pos, this.neuronRadius, color.toString(), '#111111', this.neuronRadius * 0.1);

        let text = n.output.toPrecision(1).toString();
        let size = this.neuronRadius * 0.9;
        pos.y += this.neuronRadius / 3;

        this.canvas.drawText(text, pos, this.textColor, size);
    }

    drawNeuronConnection(n1: Neuron, n2: Neuron, pos1: Vec2, pos2: Vec2)
    {
        let color = this.getNeuronColor(n2);

        this.canvas.drawLine(pos1, pos2, this.neuronRadius / 3 + (this.neuronRadius / 3) * 0.4, "black");
        this.canvas.drawLine(pos1, pos2, this.neuronRadius / 3, color.toString());
    }
}