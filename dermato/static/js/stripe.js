var stripe = Stripe('{{ STRIPE_PUBLIC_KEY }}');
const checkout_button = document.querySelector('#checkout-button');
checkout_button.addEventListener('click', event => {
stripe.redirectToCheckout({ 
    sessionId: '{{session_id}}'
    }).then(function (result) {
        
});
});
