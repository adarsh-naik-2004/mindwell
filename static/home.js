
document.addEventListener('DOMContentLoaded', (event) => {
    // Smooth scrolling for navigation links
    document.querySelectorAll('a[href^="#"]').forEach(anchor => {
        anchor.addEventListener('click', function (e) {
            e.preventDefault();
            document.querySelector(this.getAttribute('href')).scrollIntoView({
                behavior: 'smooth'
            });
        });
    });

    // Scroll reveal animation
    ScrollReveal().reveal('.scroll-reveal', {
        delay: 200,
        distance: '50px',
        duration: 1000,
        easing: 'cubic-bezier(0.5, 0, 0, 1)',
        interval: 0,
        opacity: 0,
        origin: 'bottom',
        rotate: {
            x: 0,
            y: 0,
            z: 0
        },
        scale: 1,
        cleanup: false,
        container: window.document.documentElement,
        desktop: true,
        mobile: true,
        reset: false,
        useDelay: 'always',
        viewFactor: 0.0,
        viewOffset: {
            top: 0,
            right: 0,
            bottom: 0,
            left: 0
        }
    });

    // Intersection Observer for navbar background change
    const navbar = document.querySelector('nav');
    const header = document.querySelector('header');

    const headerObserver = new IntersectionObserver((entries) => {
        entries.forEach(entry => {
            if (!entry.isIntersecting) {
                navbar.classList.add('bg-white', 'shadow-md');
            } else {
                navbar.classList.remove('bg-white', 'shadow-md');
            }
        });
    }, { threshold: 0.9 });

    headerObserver.observe(header);

    // Dynamic copyright year
    const currentYear = new Date().getFullYear();
    document.querySelector('footer p').textContent = `© ${currentYear} MindWell. All rights reserved.`;
});