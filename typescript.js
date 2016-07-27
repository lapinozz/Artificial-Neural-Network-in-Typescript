function rand(min, max) {
    if (min < 0)
        return min + Math.random() * (Math.abs(min) + max);
    else
        return min + Math.random() * max;
}
function tanh(x) {
    if (x === Infinity)
        return 1;
    else if (x === -Infinity)
        return -1;
    var y = Math.exp(2 * x);
    return (y - 1) / (y + 1);
}
function lerp(interpolation, n1, n2) {
    return (1.0 - interpolation) * n1 + interpolation * n2;
}
function lerpRGB(interpolation, rgb1, rgb2) {
    return new RGB(lerp(interpolation, rgb1.r, rgb2.r), lerp(interpolation, rgb1.g, rgb2.g), lerp(interpolation, rgb1.b, rgb2.b));
}
var RGB = (function () {
    function RGB(r, g, b) {
        if (r === void 0) { r = 0.0; }
        if (g === void 0) { g = 0.0; }
        if (b === void 0) { b = 0.0; }
        this.r = r;
        this.g = g;
        this.b = b;
    }
    RGB.prototype.toString = function () {
        return 'rgb(' + this.r.toFixed() + ',' + this.g.toFixed() + ',' + this.b.toFixed() + ')';
    };
    return RGB;
}());
var Vec2 = (function () {
    function Vec2(x, y) {
        if (x === void 0) { x = 0; }
        if (y === void 0) { y = 0; }
        this.x = x;
        this.y = y;
    }
    return Vec2;
}());
var Neuron = (function () {
    function Neuron(inputNeurons, extraWeight) {
        if (inputNeurons === void 0) { inputNeurons = null; }
        if (extraWeight === void 0) { extraWeight = 0; }
        if (inputNeurons) {
            this.inputs = inputNeurons;
            this.weights = [];
            this.deltaWeights = [];
            this.extraWeight = extraWeight;
            for (var x = 0; x < this.inputs.length + extraWeight; x++) {
                this.weights.push(rand(-1, 1));
                this.deltaWeights.push(0);
            }
        }
    }
    Neuron.prototype.computeOutput = function () {
        var sum = 0;
        for (var x = 0; x < this.inputs.length; x++)
            sum += this.inputs[x].output * this.weights[x];
        for (var x = this.inputs.length; x < this.weights.length; x++)
            sum += this.weights[x];
        this.output = Neuron.activate(sum);
    };
    Neuron.prototype.computeOutputGradiant = function (targetValue) {
        var delta = targetValue - this.output;
        this.gradient = delta * Neuron.activateAlternative(this.output);
    };
    Neuron.prototype.computeHiddenGradients = function (outputs) {
        var dow = 0.0;
        for (var x = 0; x < outputs.length; x++) {
            var index = outputs[x].inputs.indexOf(this);
            dow += outputs[x].weights[index] * outputs[x].gradient;
        }
        this.gradient = dow * Neuron.activateAlternative(this.output);
    };
    Neuron.prototype.train = function (learningRate, alpha) {
        if (learningRate === void 0) { learningRate = 0.3; }
        if (alpha === void 0) { alpha = 0.5; }
        for (var x = 0; x < this.weights.length; x++) {
            var oldDeltaWeight = this.deltaWeights[x];
            var newDeltaWeight = 
            // Individual input, magnified by the gradient and train rate:
            learningRate
                * (x < this.inputs.length ? this.inputs[x].output : 1)
                * this.gradient
                + alpha
                    * oldDeltaWeight;
            this.deltaWeights[x] = newDeltaWeight;
            this.weights[x] += newDeltaWeight;
        }
    };
    Neuron.activate = function (sum) {
        return tanh(sum);
    };
    Neuron.activateAlternative = function (sum) {
        return 1.0 - tanh(sum) * tanh(sum);
    };
    Neuron.activateStrict = function (sum) {
        return (sum > 0) ? 1 : -1;
    };
    return Neuron;
}());
var Network = (function () {
    function Network(neuronPerLayer, extraWeight) {
        this.layers = [];
        for (var neuronCount in neuronPerLayer) {
            var previousLayer = this.getOutputLayer();
            this.layers.push([]);
            for (var x = 0; x < neuronPerLayer[neuronCount]; x++)
                this.getOutputLayer().push(new Neuron(previousLayer, extraWeight ? 1 : 0));
        }
    }
    Network.prototype.getInputLayer = function () {
        if (this.layers)
            return this.layers[0];
        else
            return [];
    };
    Network.prototype.getOutputLayer = function () {
        if (this.layers)
            return this.layers[this.layers.length - 1];
        else
            return [];
    };
    Network.prototype.getLayer = function (layerIndex) {
        return this.layers[layerIndex];
    };
    Network.prototype.getLayerSize = function (layerIndex) {
        return this.getLayer(layerIndex).length;
    };
    Network.prototype.getLayerCount = function () {
        return this.layers.length;
    };
    Network.prototype.getNeuron = function (layerIndex, neuronIndex) {
        return this.layers[layerIndex][neuronIndex];
    };
    Network.prototype.activate = function (inputs) {
        for (var x in this.getLayer(0))
            this.getNeuron(0, +x).output = inputs[x];
        for (var x in this.layers) {
            if (x == '0')
                continue;
            for (var _i = 0, _a = this.getLayer(+x); _i < _a.length; _i++) {
                var n = _a[_i];
                n.computeOutput();
            }
        }
        var output = [];
        for (var _b = 0, _c = this.getOutputLayer(); _b < _c.length; _b++) {
            var n = _c[_b];
            output.push(n.output);
        }
        return output;
    };
    Network.prototype.train = function (targetOutput) {
        for (var x in this.getOutputLayer())
            this.getOutputLayer()[x].computeOutputGradiant(targetOutput[x]);
        for (var x = this.getLayerCount() - 2; x > 0; x--) {
            for (var _i = 0, _a = this.getLayer(x); _i < _a.length; _i++) {
                var n = _a[_i];
                n.computeHiddenGradients(this.getLayer(x + 1));
            }
        }
        for (var x in this.layers) {
            if (x == '0')
                continue;
            for (var _b = 0, _c = this.getLayer(+x); _b < _c.length; _b++) {
                var n = _c[_b];
                n.train();
            }
        }
    };
    return Network;
}());
var NetworkRenderer = (function () {
    function NetworkRenderer(network, canvas, pos) {
        this.verticalSpacing = 140;
        this.horizontalSpacing = 140;
        this.neuronRadius = 50;
        this.activeColor = new RGB(0, 128, 0);
        this.inactiveColor = new RGB(128, 0, 0);
        this.textColor = "white";
        this.network = network;
        this.canvas = canvas;
        this.pos = pos;
    }
    NetworkRenderer.prototype.getNeuronPosition = function (layerIndex, layerSize, neuronIndex) {
        var x = +layerIndex * this.horizontalSpacing;
        x += this.neuronRadius + this.neuronRadius * 0.1;
        var y = neuronIndex * this.verticalSpacing;
        y -= ((layerSize - 1) * this.verticalSpacing) / 2;
        return new Vec2(this.pos.x + x, this.pos.y + y);
    };
    NetworkRenderer.prototype.getNeuronColor = function (n) {
        return lerpRGB((n.output - 1.0) / -2.0, this.activeColor, this.inactiveColor);
    };
    NetworkRenderer.prototype.draw = function () {
        for (var layerIndex in this.network.layers) {
            var previousLayer = this.network.getLayer(+layerIndex - 1);
            var layer = this.network.getLayer(+layerIndex);
            for (var n in layer) {
                var pos1 = this.getNeuronPosition(+layerIndex, layer.length, +n);
                if (!layer[n].inputs)
                    continue;
                for (var i in layer[n].inputs) {
                    var pos2 = this.getNeuronPosition(+layerIndex - 1, previousLayer.length, +i);
                    this.drawNeuronConnection(layer[n], layer[n].inputs[i], pos1, pos2);
                }
            }
        }
        for (var layerIndex in this.network.layers) {
            var layer = this.network.getLayer(+layerIndex);
            for (var n in layer) {
                var pos = this.getNeuronPosition(+layerIndex, layer.length, +n);
                this.drawNeuron(layer[n], pos);
            }
        }
    };
    NetworkRenderer.prototype.drawNeuron = function (n, pos) {
        var color = this.getNeuronColor(n);
        this.canvas.drawCircle(pos, this.neuronRadius, color.toString(), '#111111', this.neuronRadius * 0.1);
        var text = n.output.toPrecision(1).toString();
        var size = this.neuronRadius * 0.9;
        pos.y += this.neuronRadius / 3;
        this.canvas.drawText(text, pos, this.textColor, size);
    };
    NetworkRenderer.prototype.drawNeuronConnection = function (n1, n2, pos1, pos2) {
        var color = this.getNeuronColor(n2);
        this.canvas.drawLine(pos1, pos2, this.neuronRadius / 3 + (this.neuronRadius / 3) * 0.4, "black");
        this.canvas.drawLine(pos1, pos2, this.neuronRadius / 3, color.toString());
    };
    return NetworkRenderer;
}());
var Canvas = (function () {
    function Canvas(id, size) {
        if (size === void 0) { size = null; }
        this.canvas = document.getElementById(id);
        if (size == null)
            size = new Vec2(document.body.clientWidth, document.body.clientHeight);
        this.canvas.width = size.x;
        this.canvas.height = size.y;
        this.ctx = this.canvas.getContext("2d");
    }
    Canvas.prototype.getWidth = function () {
        return this.canvas.width;
    };
    Canvas.prototype.getHeight = function () {
        return this.canvas.height;
    };
    Canvas.prototype.drawCircle = function (pos, radius, fillStyle, strokeStyle, lineWidth) {
        if (lineWidth === void 0) { lineWidth = 10; }
        this.ctx.beginPath();
        this.ctx.arc(pos.x, pos.y, radius, 0, 2 * Math.PI, false);
        this.ctx.fillStyle = fillStyle;
        this.ctx.fill();
        this.ctx.lineWidth = lineWidth;
        this.ctx.strokeStyle = strokeStyle;
        this.ctx.stroke();
    };
    Canvas.prototype.drawLine = function (start, end, lineWidth, strokeStyle) {
        this.ctx.beginPath();
        this.ctx.moveTo(start.x, start.y);
        this.ctx.lineTo(end.x, end.y);
        this.ctx.lineWidth = lineWidth;
        this.ctx.strokeStyle = strokeStyle;
        this.ctx.stroke();
    };
    Canvas.prototype.drawText = function (text, pos, fillStyle, size, textAlign, font) {
        if (textAlign === void 0) { textAlign = "center"; }
        if (font === void 0) { font = "Arial"; }
        this.ctx.textAlign = textAlign;
        // this.ctx.font = (size + 1) + "px " + font;
        // this.ctx.strokeStyle = "black";
        // this.ctx.strokeText(text, pos.x, pos.y);
        this.ctx.font = size + "px " + font;
        this.ctx.fillStyle = fillStyle;
        this.ctx.fillText(text, pos.x, pos.y);
    };
    Canvas.prototype.clear = function (fillType) {
        if (fillType === void 0) { fillType = "white"; }
        this.ctx.fillStyle = fillType;
        this.ctx.fillRect(0, 0, this.getWidth(), this.getHeight());
    };
    return Canvas;
}());
//[[-1, -1], [-1]], [[1, -1], [-1]], [[-1, 1], [-1]], [[1, 1], [1]]
var trainSetAND = [];
trainSetAND.push([[-1, -1], [-1]]);
trainSetAND.push([[1, -1], [-1]]);
trainSetAND.push([[-1, 1], [-1]]);
trainSetAND.push([[1, 1], [1]]);
//[[-1, -1, -1], [1, -1, 1], [-1, 1, 1], [1, 1, 1]]
var trainSetOR = [];
trainSetOR.push([[-1, -1], [-1]]);
trainSetOR.push([[1, -1], [1]]);
trainSetOR.push([[-1, 1], [1]]);
trainSetOR.push([[1, 1], [1]]);
//[[[-1, -1], [-1]], [[1, -1], [1]], [[-1, 1], [1]], [[1, 1], [-1]]]
var trainSetXOR = [];
trainSetXOR.push([[-1, -1], [-1]]);
trainSetXOR.push([[1, -1], [1]]);
trainSetXOR.push([[-1, 1], [1]]);
trainSetXOR.push([[1, 1], [-1]]);
var trainSets = [trainSetAND, trainSetOR, trainSetXOR];
var network = new Network([2, 2, 1], true);
var canvas;
var renderer;
var speedSlider;
var speedDiv;
var pauseCheckbox;
window.onload = function () {
    var size = new Vec2(document.body.clientWidth, document.body.clientHeight / 2);
    canvas = new Canvas('cnvs', size);
    var pos = new Vec2(0, canvas.getHeight() / 2);
    renderer = new NetworkRenderer(network, canvas, pos);
    speedSlider = document.getElementById("speedSlider");
    speedDiv = document.getElementById("speedDiv");
    pauseCheckbox = document.getElementsByName("pauseButton")[0];
    window.setTimeout(updateNetwork, 1000 / 60);
    window.setTimeout(updateGui, 1000 / 60);
};
function updateGui() {
    window.setTimeout(updateGui, 100);
    speedDiv.innerHTML = (1000 / +speedSlider.value).toFixed(2) + " FPS";
}
var trainSetIndex = 0;
var step = 0;
function updateNetwork() {
    var timeout = pauseCheckbox.checked ? 1 : +speedSlider.value;
    window.setTimeout(updateNetwork, timeout);
    if (pauseCheckbox.checked && step == 0)
        return;
    if (step != 0)
        step--;
    var radioButtons = document.getElementsByName("trainType");
    var checkedButton;
    var trainSet;
    for (var rd in radioButtons) {
        if (radioButtons[rd].checked) {
            checkedButton = radioButtons[rd];
            trainSet = trainSets[rd];
            break;
        }
    }
    var set = trainSet[trainSetIndex];
    trainSetIndex++;
    if (trainSetIndex >= trainSet.length)
        trainSetIndex = 0;
    var outputs = network.activate(set[0]);
    network.train(set[1]);
    canvas.clear();
    renderer.draw();
    var textSize = 30;
    var textPos = new Vec2(700, canvas.getHeight() / 2);
    textPos.y += textSize / 3;
    var text;
    var color;
    var textInput1 = set[0][0] == 1 ? 1 : 0;
    var textInput2 = set[0][1] == 1 ? 1 : 0;
    var textOutput = set[1][0] == 1 ? 1 : 0;
    text = textInput1 + " " + checkedButton.value + " " + textInput2 + " = " + textOutput + "\n";
    var output = Neuron.activateStrict(outputs[0]);
    if (output == set[1][0]) {
        text += "Correct";
        color = "green";
    }
    else {
        text += "Incorrect";
        color = "red";
    }
    canvas.drawText(text, textPos, color, textSize);
}
function stepButton() {
    step++;
}
//# sourceMappingURL=typescript.js.map