/**
 * GSAP Animation Controller for Epilepsy CDSS
 * Built using GSAP Core, Timeline, ScrollTrigger, and Utils
 * Includes Preloader, Kinetic Text Animations, Scroll-Linked Reveals & Interactive Signals
 */

(function () {
  "use strict";

  // Accessibility: Check reduced motion preference
  const prefersReducedMotion = window.matchMedia("(prefers-reduced-motion: reduce)").matches;

  // Initialize on DOM Ready
  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", initAppAnimations);
  } else {
    initAppAnimations();
  }

  function initAppAnimations() {
    if (typeof gsap === "undefined") {
      console.warn("GSAP not loaded. Skipping animations.");
      const preloader = document.getElementById("app-preloader");
      if (preloader) preloader.style.display = "none";
      return;
    }

    // Register ScrollTrigger plugin if present
    if (typeof ScrollTrigger !== "undefined") {
      gsap.registerPlugin(ScrollTrigger);
    }

    // Global GSAP defaults
    gsap.defaults({
      duration: 0.6,
      ease: "power2.out"
    });

    if (prefersReducedMotion) {
      const preloader = document.getElementById("app-preloader");
      if (preloader) preloader.style.display = "none";
      gsap.set("[data-gsap-reveal], .feature-editorial-item, .report-metric-card, .table-wrapper", {
        autoAlpha: 1,
        y: 0,
        scale: 1
      });
      initNumberCounters(true);
      initEegWaveform();
      return;
    }

    // 1. Run Preloader Animation Sequence
    initPreloader(() => {
      // 2. Trigger Page Entrance & Text Animations once Preloader clears
      runPageEntrance();
      initScrollAnimations();
      initMagneticButtons();
      initMarquee();
      initNumberCounters(false);
      initEegWaveform();
    });
  }

  /**
   * Preloader Sequence: Simulates neural signal calibration & curtain reveal
   */
  function initPreloader(onCompleteCallback) {
    const preloader = document.getElementById("app-preloader");
    const counterEl = document.getElementById("preloader-pct");
    const barEl = document.getElementById("preloader-bar");
    const pulsePath = document.getElementById("preloader-wave-pulse");

    if (!preloader) {
      if (typeof onCompleteCallback === "function") onCompleteCallback();
      return;
    }

    // Check if user has already seen the preloader this session
    const hasSeenPreloader = sessionStorage.getItem("epilepsy_cdss_preloader_seen");
    const animDuration = hasSeenPreloader ? 0.45 : 1.1;

    const preloaderTl = gsap.timeline({
      onComplete: () => {
        sessionStorage.setItem("epilepsy_cdss_preloader_seen", "true");
        gsap.to(preloader, {
          yPercent: -100,
          duration: 0.65,
          ease: "power4.inOut",
          onComplete: () => {
            preloader.style.display = "none";
            if (typeof onCompleteCallback === "function") onCompleteCallback();
          }
        });
      }
    });

    // Animate wave path stroke drawing
    if (pulsePath) {
      preloaderTl.fromTo(
        pulsePath,
        { strokeDashoffset: 400 },
        { strokeDashoffset: 0, duration: animDuration, ease: "power2.inOut" },
        0
      );
    }

    // Animate progress bar & numeric counter
    const progressObj = { value: 0 };
    preloaderTl.to(
      progressObj,
      {
        value: 100,
        duration: animDuration,
        ease: "power2.out",
        onUpdate: () => {
          const currentPct = Math.round(progressObj.value);
          if (counterEl) counterEl.textContent = currentPct + "%";
          if (barEl) barEl.style.width = currentPct + "%";
        }
      },
      0
    );

    // Subtle breathing pulse on the brand icon
    preloaderTl.to(
      ".preloader-brand-icon",
      {
        scale: 1.06,
        duration: animDuration * 0.5,
        yoyo: true,
        repeat: 1,
        ease: "sine.inOut"
      },
      0
    );

    // Fade out preloader inner content just before curtain lift
    preloaderTl.to(".preloader-content", {
      y: -20,
      autoAlpha: 0,
      duration: 0.25,
      ease: "power2.in"
    });
  }

  /**
   * Page Entrance Timeline: Staggered typography and hero reveals
   */
  function runPageEntrance() {
    const entranceTl = gsap.timeline({ defaults: { ease: "power3.out" } });

    // Top navigation reveal
    entranceTl.from(".top-nav", {
      y: -30,
      autoAlpha: 0,
      duration: 0.6,
      ease: "power2.out"
    }, 0);

    // Home / Dashboard Impact Hero
    if (document.querySelector(".hero-impact-section")) {
      entranceTl
        .from(".hero-bg-visual", {
          scale: 1.15,
          autoAlpha: 0,
          duration: 1.0,
          ease: "power2.out"
        }, 0.1)
        .from(".hero-reveal", {
          y: 35,
          autoAlpha: 0,
          duration: 0.65,
          stagger: 0.08,
          ease: "power4.out"
        }, 0.2)
        .from(".hero-proof-item", {
          y: 15,
          autoAlpha: 0,
          stagger: 0.06,
          duration: 0.5,
          ease: "power2.out"
        }, 0.45);
    } 
    // Standard Page Header & Title
    else if (document.querySelector(".page-header")) {
      entranceTl
        .from(".page-header .hero-badge-pill", {
          y: -10,
          autoAlpha: 0,
          duration: 0.4
        }, 0.1)
        .from(".page-header h1, .page-header .page-title", {
          y: 25,
          autoAlpha: 0,
          duration: 0.6,
          ease: "power4.out"
        }, 0.2)
        .from(".page-header .page-subtitle", {
          y: 15,
          autoAlpha: 0,
          duration: 0.5
        }, 0.3);
    }

    // Report Dossier Hero Bar
    if (document.querySelector(".report-hero-bar")) {
      entranceTl
        .from(".report-hero-bar", {
          y: 20,
          autoAlpha: 0,
          duration: 0.6,
          ease: "power3.out"
        }, 0.15)
        .from(".report-metric-card", {
          y: 20,
          autoAlpha: 0,
          stagger: 0.08,
          duration: 0.55,
          ease: "power3.out"
        }, 0.3);
    }
  }

  /**
   * Scroll-Triggered Batch Animations
   */
  function initScrollAnimations() {
    if (typeof ScrollTrigger === "undefined") return;

    // Feature editorial items stagger
    ScrollTrigger.batch(".feature-editorial-item", {
      interval: 0.1,
      batchMax: 4,
      onEnter: (batch) => {
        gsap.fromTo(
          batch,
          { y: 35, autoAlpha: 0 },
          { y: 0, autoAlpha: 1, stagger: 0.1, duration: 0.65, ease: "power3.out", overwrite: "auto" }
        );
      },
      once: true
    });

    // Typographic stats row stagger
    ScrollTrigger.batch(".stat-editorial-box, .stat-item-clean", {
      interval: 0.08,
      batchMax: 4,
      onEnter: (batch) => {
        gsap.fromTo(
          batch,
          { y: 25, autoAlpha: 0 },
          { y: 0, autoAlpha: 1, stagger: 0.08, duration: 0.6, ease: "power3.out", overwrite: "auto" }
        );
      },
      once: true
    });

    // Section headings & general reveal-on-scroll elements
    const sectionHeadings = gsap.utils.toArray(".section-title, .reveal-on-scroll");
    sectionHeadings.forEach((heading) => {
      gsap.from(heading, {
        scrollTrigger: {
          trigger: heading,
          start: "top 88%"
        },
        y: 25,
        autoAlpha: 0,
        duration: 0.6,
        ease: "power3.out"
      });
    });

    // Table rows stagger reveal on entering viewport
    const reportTables = document.querySelectorAll(".report-table tbody");
    reportTables.forEach((tbody) => {
      const rows = tbody.querySelectorAll("tr");
      if (rows.length > 0) {
        ScrollTrigger.create({
          trigger: tbody,
          start: "top 85%",
          once: true,
          onEnter: () => {
            gsap.fromTo(
              rows,
              { autoAlpha: 0, y: 12 },
              { autoAlpha: 1, y: 0, stagger: 0.03, duration: 0.4, ease: "power2.out" }
            );
          }
        });
      }
    });
  }

  /**
   * Magnetic Button Hover Effects
   */
  function initMagneticButtons() {
    const magneticButtons = document.querySelectorAll(".btn.primary, .btn-arrow-hover");
    magneticButtons.forEach((btn) => {
      btn.addEventListener("mousemove", (e) => {
        const rect = btn.getBoundingClientRect();
        const x = e.clientX - rect.left - rect.width / 2;
        const y = e.clientY - rect.top - rect.height / 2;
        gsap.to(btn, {
          x: x * 0.18,
          y: y * 0.18,
          duration: 0.3,
          ease: "power2.out"
        });
      });

      btn.addEventListener("mouseleave", () => {
        gsap.to(btn, {
          x: 0,
          y: 0,
          duration: 0.5,
          ease: "elastic.out(1.2, 0.4)"
        });
      });
    });
  }

  /**
   * Infinite Marquee Track (Smooth Loop)
   */
  function initMarquee() {
    const track = document.querySelector(".marquee-track");
    if (!track) return;

    gsap.to(track, {
      xPercent: -50,
      repeat: -1,
      duration: 22,
      ease: "none"
    });
  }

  /**
   * Animated Number Counters with GSAP
   */
  function initNumberCounters(immediate) {
    const counterElements = document.querySelectorAll("[data-counter-target]");
    counterElements.forEach((el) => {
      const targetVal = parseFloat(el.getAttribute("data-counter-target")) || 0;
      const decimals = parseInt(el.getAttribute("data-counter-decimals") || "0", 10);
      const suffix = el.getAttribute("data-counter-suffix") || "";
      const prefix = el.getAttribute("data-counter-prefix") || "";

      if (immediate) {
        el.textContent = prefix + targetVal.toFixed(decimals) + suffix;
        return;
      }

      const counterObj = { val: 0 };
      
      if (typeof ScrollTrigger !== "undefined") {
        ScrollTrigger.create({
          trigger: el,
          start: "top 90%",
          once: true,
          onEnter: () => {
            gsap.to(counterObj, {
              val: targetVal,
              duration: 1.3,
              ease: "power2.out",
              onUpdate: () => {
                el.textContent = prefix + counterObj.val.toFixed(decimals) + suffix;
              }
            });
          }
        });
      } else {
        gsap.to(counterObj, {
          val: targetVal,
          duration: 1.3,
          ease: "power2.out",
          onUpdate: () => {
            el.textContent = prefix + counterObj.val.toFixed(decimals) + suffix;
          }
        });
      }
    });
  }

  /**
   * Interactive Canvas EEG Waveforms (60fps simulation)
   */
  function initEegWaveform() {
    const canvases = document.querySelectorAll(".eeg-channel-canvas, .waveform-canvas-interactive");
    if (!canvases.length) return;

    canvases.forEach((canvas, canvasIndex) => {
      const ctx = canvas.getContext("2d");
      if (!ctx) return;

      let animationFrameId;
      let offset = canvasIndex * 40;

      function resize() {
        const dpr = window.devicePixelRatio || 1;
        const rect = canvas.getBoundingClientRect();
        canvas.width = (rect.width || 300) * dpr;
        canvas.height = (rect.height || 60) * dpr;
        ctx.scale(dpr, dpr);
      }

      resize();
      window.addEventListener("resize", resize);

      function draw() {
        const rect = canvas.getBoundingClientRect();
        const width = rect.width || 300;
        const height = rect.height || 60;
        const midY = height / 2;

        ctx.clearRect(0, 0, width, height);

        // Waveform stroke styling
        ctx.beginPath();
        ctx.strokeStyle = canvas.getAttribute("data-waveform-color") || "#ff4d8b";
        ctx.lineWidth = 1.8;
        ctx.lineJoin = "round";
        ctx.lineCap = "round";

        const points = 80;
        const step = width / points;

        for (let i = 0; i <= points; i++) {
          const x = i * step;
          const freq1 = Math.sin((i * 0.18) + (offset * 0.05));
          const freq2 = Math.cos((i * 0.35) + (offset * 0.08)) * 0.5;
          const spike = (i % 18 === 0) ? (Math.random() > 0.4 ? 14 : -14) : 0;
          const y = midY + (freq1 * 12) + (freq2 * 6) + spike;

          if (i === 0) {
            ctx.moveTo(x, y);
          } else {
            ctx.lineTo(x, y);
          }
        }

        ctx.stroke();
        offset += 0.75;
        animationFrameId = requestAnimationFrame(draw);
      }

      draw();
    });
  }
})();
