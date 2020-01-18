$(document).ready(function() {
  
  bsCustomFileInput.init();

  const spinner = document.getElementById("loading-spinner");
  const uploadButton = document.getElementById("submit");

  $('form').submit(function() {

    spinner.classList.add('show');
    uploadButton.classList.add('disabled');
    uploadButton.textContent = 'Processing...';
  });

});
