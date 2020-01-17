$(document).ready(function() {
  
  bsCustomFileInput.init();

  const spinner = document.getElementById("loading-spinner");
  const uploadButton = document.getElementById("upload");

  $('#upload').click(function() {

    spinner.classList.add('show');
    uploadButton.disabled = true;

    var form_data = new FormData($('form')[0]);
    // var form_data = new FormData();
    // var fileToUpload = $('#customFile').prop('files')[0];
    // form_data.append('file', fileToUpload);

    var request = $.ajax({
      url: "separate",
      method: "POST",
      data: form_data,
      dataType: "json"
    });
     
    request.done(function( data ) {
      if (data.hasOwnProperty('error')) {
        $('#errors').fadeIn();
        $('#errors').html(data['error']);
        return;
      }

      // TODO: Happy path
      alert("done (success)")

      // Clear out any errors:
      $('#errors').hide();
    });
     
    request.fail(function( jqXHR, textStatus ) {
      alert(textStatus);
      alert( "Request failed: " + textStatus );
    });

    request.always(function( msg ) { // Cleanup!
      uploadButton.disabled = false;
      spinner.classList.remove('show');
    });


  });

});
