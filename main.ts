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

let pauseCheckbox: HTMLInputElement;

window.onload = () => 
{
    let size = new Vec2(document.body.clientWidth, document.body.clientHeight / 2);
    canvas = new Canvas('cnvs', size);

    let pos = new Vec2(0, canvas.getHeight() / 2);
    renderer = new NetworkRenderer(network, canvas, pos);

    speedSlider = <HTMLInputElement>document.getElementById("speedSlider");
    speedDiv = <HTMLDivElement>document.getElementById("speedDiv");

    pauseCheckbox = <HTMLInputElement>document.getElementsByName("pauseButton")[0];

    window.setTimeout(updateNetwork, 1000 / 60);
    window.setTimeout(updateGui, 1000 / 60);
}

function updateGui()
{
    window.setTimeout(updateGui, 100);
    speedDiv.innerHTML = (1000 / +speedSlider.value).toFixed(2) + " FPS";
}

let trainSetIndex = 0;
let step = 0;
function updateNetwork()
{
    let timeout = pauseCheckbox.checked ? 1 : +speedSlider.value;
    window.setTimeout(updateNetwork, timeout);

    if(pauseCheckbox.checked && step == 0)
        return;

    if(step != 0)
        step--;

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

function stepButton()
{
    step++;
}