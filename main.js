document.querySelector('form').addEventListener('submit', function (event) {
    const inputs = document.querySelectorAll('input');
    let isValid = true;

    inputs.forEach(input => {
        if (input.value === '' || parseFloat(input.value) <= 0) {
            alert(`${input.name} is invalid. Please provide a valid value.`);
            isValid = false;
        }
    });

    if (!isValid) {
        event.preventDefault();
    } else {
        // Show loading spinner
        const button = document.querySelector('button');
        button.textContent = "Processing...";
        button.disabled = true;
    }
});
