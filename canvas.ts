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