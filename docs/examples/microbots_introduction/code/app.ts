// app.ts — A simple greeting function with a syntax error
function greet(name: string) {
    const message: string = `Hello, ${name}!`
    console.log(message)
}

// Missing closing parenthesis on the type annotation
function add(a: number, b: number: number {
    return a + b;
}

greet("Microbots");
console.log(add(2, 3));
