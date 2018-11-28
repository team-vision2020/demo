// var detectModels;
// var inverseModels = {}
$( document ).ready(function() {
    disableSubmit();
    // Load model
    // modelDetect = await tf.loadModel('./identifier.json');
    $("#file").change(function() {
        readURL(this);
        enableSubmit();
    });
    $('#image-preview').hide();
    $('#submit-btn').click(processInvert);
});

const onSelection = () => {
    enableSubmit();
}

const disableSubmit = () => {
    $('#submit-btn').css({
        pointerEvents: 'none',
        color: 'gray'
    });
}

const enableSubmit = () => {
    $('#submit-btn').css({
        pointerEvents: 'auto',
        color: 'black'
    });
    $('#image-wrap').css({
        backgroundColor: 'transparent'
    });
    $('#image-preview').show();
}

/* snippet adapted from https://stackoverflow.com/questions/4459379/preview-an-image-before-it-is-uploaded */
const readURL = input => {
    if (input.files && input.files[0]) {
      var reader = new FileReader();
  
      reader.onload = function(e) {
        $('#image-preview').attr('src', e.target.result);
      }
  
      reader.readAsDataURL(input.files[0]);
    }
}

const processInvert = () => {
    const file_input = document.getElementById('file');
    const form = new FormData();
    form.append("image", file_input.files[0]); // it is being loaded properly
    $.ajax({
            url: "/process",
            method: "POST",
            data: form,
            processData: false,
            contentType: false,
            success: function(result){ /* Dict of filter identified, image url */
                showResults(result);
            },
            error: function(er){
                console.log("Ajax fail");
            }
    });
    /* Loading state here */
};

const showResults = ({img_url, filter}) => {
    const capital = filter.substring(0,1).toUpperCase() + filter.substring(1);
    $('#image-preview').attr('src', `/static/uploads/${img_url}`);
    $('#detected-filter').html(`Detected: ${capital}`);
};
  