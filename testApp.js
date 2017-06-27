var app = {
    points : [],
    startingPoint : {x:0, y:0},
    canvas : undefined,
    context : undefined,
    init : function(numPoints, canvas) {
        this.canvas = canvas;
        this.context = this.canvas.getContext("2d");
        this.startingPoint = {x: Math.random() * this.canvas.width, y: Math.random() * this.canvas.height};
        this.points = [];
        for (i = 0; i<numPoints; i++) {
            var a = Math.random() * this.canvas.width;
            var b = Math.random() * this.canvas.height;
            this.points.push({x:a, y:b});
        }
        this.drawStartingPoints();
    },
    start : function() {
        this.placeNextPoint();
    },
    reset : function() {
        this.context.clearRect(0, 0, this.canvas.width, canvas.height);
        this.init(this.points.length, this.canvas);
    },
    placeNextPoint : function() {
        var destination = Math.floor(Math.random() * this.points.length);
        var destPoint = this.points[destination];
        var x = (this.startingPoint.x + destPoint.x) / 2;
        var y = (this.startingPoint.y + destPoint.y) / 2;
        this.drawPoint(x, y, 1, "red");
        this.startingPoint.x = x;
        this.startingPoint.y = y;
        this.placeNextPoint();
    },
    drawStartingPoints : function() {
        for (i=0; i<this.points.length; i++) {
            var point = this.points[i];
            this.drawPoint(point.x, point.y, 4, "black");
        }
        this.drawPoint(this.startingPoint.x, this.startingPoint.y, 4, "blue");
    },
    drawPoint : function (x, y, w, color) {
        this.context.beginPath();
        this.context.arc(x, y, w, 0, 2 * Math.PI, true);
        this.context.fillStyle = color;
        this.context.fill();
    }
}