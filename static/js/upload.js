// Wait for the DOM to load
document.addEventListener('DOMContentLoaded', function() {
    // Get the input element
    const fileInput = document.getElementById('txtFileUpload');
  
    // Attach an event listener to the input element
    fileInput.addEventListener('change', function(event) {
      // Get the file object
      const file = event.target.files[0];
  
      // Create a FileReader object
      const reader = new FileReader();
  
      // Attach an event listener to the FileReader object
      reader.addEventListener('load', function(event) {
        // Get the CSV file contents as a string
        const csv = event.target.result;
  
        // Split the string into rows
        const rows = csv.split('\n');
  
        // Update the HTML with the row and column counts
        const countOfRows = rows.length;
        const countOfColumns = rows[0].split(',').length;
        document.querySelector('#statOutPut span[data-bind="text: countOfRows"]').textContent = countOfRows;
        document.querySelector('#statOutPut span[data-bind="text: countOfColumns"]').textContent = countOfColumns;
      });
  
      // Read the file as text
      reader.readAsText(file);
    });
  });
  