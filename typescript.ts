function rand(min: number, max: number)
{
    if(min < 0)
        return min + Math.random() * (Math.abs(min) + max);
    else
        return min + Math.random() * max;
}

function tanh(x: number)
{
    if(x === Infinity)
        return 1;
    else if(x === -Infinity)
        return -1;

    let y = Math.exp(2 * x);
    return (y - 1) / (y + 1);
}

function lerp(interpolation: number, n1: number, n2: number)
{
    return (1.0 - interpolation) * n1 + interpolation * n2;
}

function lerpRGB(interpolation: number, rgb1: RGB, rgb2: RGB)
{
    return new RGB(lerp(interpolation, rgb1.r, rgb2.r), lerp(interpolation, rgb1.g, rgb2.g), lerp(interpolation, rgb1.b, rgb2.b))
}

class RGB
{
    r: number;
    g: number;
    b: number;

    constructor(r: number = 0.0, g: number = 0.0, b: number = 0.0)
    {
        this.r = r;
        this.g = g;
        this.b = b;
    }

    toString()
    {
        return 'rgb(' + this.r.toFixed() + ',' + this.g.toFixed() + ',' + this.b.toFixed() + ')';
    }
}

class Vec2
{
    x: number;
    y: number;

    constructor(x: number = 0, y: number = 0)
    {
        this.x = x;
        this.y = y;
    }
}

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

class Canvas
{
    canvas: HTMLCanvasElement;
    ctx: CanvasRenderingContext2D;

    constructor(id: string, size: Vec2 = null)
    {
        this.canvas = <HTMLCanvasElement>document.getElementById(id);

        if(size == null)
            size = new Vec2(document.body.clientWidth, document.body.clientHeight);

        this.canvas.width = size.x;
        this.canvas.height = size.y;

        this.ctx = this.canvas.getContext("2d");
    }

    getWidth()
    {
        return this.canvas.width;
    }

    getHeight()
    {
        return this.canvas.height;
    }

    drawCircle(pos: Vec2, radius: number, fillStyle: string, strokeStyle: string, lineWidth: number = 10)
    {
        this.ctx.beginPath();
        this.ctx.arc(pos.x, pos.y, radius, 0, 2 * Math.PI, false);
        this.ctx.fillStyle = fillStyle
        this.ctx.fill();
        this.ctx.lineWidth = lineWidth;
        this.ctx.strokeStyle = strokeStyle;
        this.ctx.stroke();
    }

    drawLine(start: Vec2, end: Vec2, lineWidth: number, strokeStyle: string)
    {
        this.ctx.beginPath();
        this.ctx.moveTo(start.x, start.y);
        this.ctx.lineTo(end.x, end.y);

        this.ctx.lineWidth = lineWidth;
        this.ctx.strokeStyle = strokeStyle;
        this.ctx.stroke();
    }

    drawText(text: string, pos: Vec2, fillStyle: string, size: number, textAlign: string = "center", font: string = "Arial")
    {
        this.ctx.textAlign = textAlign;
        // this.ctx.font = (size + 1) + "px " + font;
        // this.ctx.strokeStyle = "black";
        // this.ctx.strokeText(text, pos.x, pos.y);

        this.ctx.font = size + "px " + font;
        this.ctx.fillStyle = fillStyle;
        this.ctx.fillText(text, pos.x, pos.y);
    }

    clear(fillType: string = "white")
    {
        this.ctx.fillStyle = fillType;
        this.ctx.fillRect(0, 0, this.getWidth(), this.getHeight());
    }
}

//[[-1, -1], [-1]], [[1, -1], [-1]], [[-1, 1], [-1]], [[1, 1], [1]]
let trainSetAND = [];
trainSetAND.push([[-1, -1], [-1]]);
trainSetAND.push([[1, -1], [-1]]);
trainSetAND.push([[-1, 1], [-1]]);
trainSetAND.push([[1, 1], [1]]);

//[[-1, -1, -1], [1, -1, 1], [-1, 1, 1], [1, 1, 1]]
let trainSetOR = [];
trainSetOR.push([[-1, -1], [-1]]);
trainSetOR.push([[1, -1], [1]]);
trainSetOR.push([[-1, 1], [1]]);
trainSetOR.push([[1, 1], [1]]);

//[[[-1, -1], [-1]], [[1, -1], [1]], [[-1, 1], [1]], [[1, 1], [-1]]]
let trainSetXOR = [];
trainSetXOR.push([[-1, -1], [-1]]);
trainSetXOR.push([[1, -1], [1]]);
trainSetXOR.push([[-1, 1], [1]]);
trainSetXOR.push([[1, 1], [-1]]);

let trainSets = [trainSetAND, trainSetOR, trainSetXOR];

let network = new Network([2, 2, 1], true);
let canvas: Canvas;
let renderer: NetworkRenderer;

let speedSlider: HTMLInputElement;
let speedDiv: HTMLDivElement;

window.onload = () => 
{
    let size = new Vec2(document.body.clientWidth, document.body.clientHeight / 2);
    canvas = new Canvas('cnvs', size);

    let pos = new Vec2(0, canvas.getHeight() / 2);
    renderer = new NetworkRenderer(network, canvas, pos);

    speedSlider = <HTMLInputElement>document.getElementById("speedSlider");
    speedDiv = <HTMLDivElement>document.getElementById("speedDiv");

    window.setTimeout(updateNetwork, 1000 / 60);
    window.setTimeout(updateGui, 1000 / 60);
}

function updateGui()
{
    window.setTimeout(updateGui, 100);
    speedDiv.innerHTML = (1000 / +speedSlider.value).toFixed(2) + " FPS";
}

let trainSetIndex = 0;
function updateNetwork()
{
    window.setTimeout(updateNetwork, +speedSlider.value);

    let radioButtons = document.getElementsByName("trainType");
    let checkedButton: HTMLInputElement;
    let trainSet;

    for(let rd in radioButtons)
    {
        if((<HTMLInputElement>radioButtons[rd]).checked)
        {
            checkedButton = <HTMLInputElement>radioButtons[rd];
            trainSet = trainSets[rd];

            break;
        }
    }

    let set = trainSet[trainSetIndex];

    trainSetIndex++;
    if(trainSetIndex >= trainSet.length)
        trainSetIndex = 0;

    let outputs = network.activate(set[0]);
    network.train(set[1]);

    canvas.clear();
    renderer.draw();

    let textSize = 30;
    let textPos = new Vec2(700, canvas.getHeight() / 2);
    textPos.y += textSize / 3;

    let text: string;
    let color: string;

    let textInput1 = set[0][0] == 1 ? 1 : 0;
    let textInput2 = set[0][1] == 1 ? 1 : 0;
    let textOutput = set[1][0] == 1 ? 1 : 0;

    text = textInput1 + " " + checkedButton.value + " " + textInput2 + " = " + textOutput + "\n"; 
    
    let output = Neuron.activateStrict(outputs[0]);

    if(output == set[1][0])
    {
        text += "Correct";
        color = "green";
    }
    else
    {
        text += "Incorrect";
        color = "red";
    }
    
    canvas.drawText(text, textPos, color, textSize);
}