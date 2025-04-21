// Main JavaScript file for the Hospital & Patient Registry application

document.addEventListener('DOMContentLoaded', function() {
    // Handle sidebar button clicks on the main page
    const registerHospitalBtn = document.getElementById('registerHospitalBtn');
    const registerPatientBtn = document.getElementById('registerPatientBtn');
    const viewDataBtn = document.getElementById('viewDataBtn');
    
    // Add event listeners if these buttons exist (i.e., on the main page)
    if (registerHospitalBtn) {
        registerHospitalBtn.addEventListener('click', function() {
            window.location.href = '/register_hospital';
        });
    }
    
    if (registerPatientBtn) {
        registerPatientBtn.addEventListener('click', function() {
            window.location.href = '/register_patient';
        });
    }
    
    if (viewDataBtn) {
        viewDataBtn.addEventListener('click', function() {
            window.location.href = '/view_data';
        });
    }
    
    // Auto-hide alerts after 5 seconds
    const alerts = document.querySelectorAll('.alert');
    alerts.forEach(function(alert) {
        setTimeout(function() {
            alert.classList.add('fade');
            setTimeout(function() {
                alert.style.display = 'none';
            }, 500);
        }, 5000);
    });
    
    // Enhance form validations
    const forms = document.querySelectorAll('.needs-validation');
    
    forms.forEach(function(form) {
        form.addEventListener('next', function(event) {
            if (!form.checkValidity()) {
                event.preventDefault();
                event.stopPropagation();
            }
            window.location.href = '/recommend_hospital';    
            form.classList.add('was-validated');
        }, false);
    });
    
    // Pain score slider value display
    const painSlider = document.getElementById('pain_score');
    if (painSlider) {
        painSlider.addEventListener('input', function() {
            this.nextElementSibling.value = this.value;
        });
    }
});