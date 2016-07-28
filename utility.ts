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