//--------------------------------------------------------------------------------------
// RawGameControllerUWP.h
//
// Advanced Technology Group (ATG)
// Copyright (C) Microsoft Corporation. All rights reserved.
//--------------------------------------------------------------------------------------

#pragma once

#include "DeviceResources.h"
#include "StepTimer.h"
#include <Windows.Gaming.Input.h>
#include <collection.h>


// A basic sample implementation that creates a D3D11 device and
// provides a render loop.
class Sample : public DX::IDeviceNotify
{
public:

    Sample();

    // Initialization and management
    void Initialize(IUnknown* window, int width, int height, DXGI_MODE_ROTATION rotation);

    // Basic render loop
    void Tick();

    // IDeviceNotify
    virtual void OnDeviceLost() override;
    virtual void OnDeviceRestored() override;

    // Messages
    void OnActivated();
    void OnDeactivated();
    void OnSuspending();
    void OnResuming();
    void OnWindowSizeChanged(int width, int height, DXGI_MODE_ROTATION rotation);
    void ValidateDevice();

    // Properties
    void GetDefaultSize(int& width, int& height) const;

private:

    void Update(DX::StepTimer const& timer);
    void Render();
	void UpdateWheel();
	void UpdateController();
	void DrawWheel(DirectX::XMFLOAT2 startPosition);
    void RefreshControllerInfo();

    void Clear();

    void CreateDeviceDependentResources();
    void CreateWindowSizeDependentResources();

	Windows::Globalization::Calendar^ cal;

    // Render objects.
    std::unique_ptr<DirectX::SpriteBatch>   m_spriteBatch;
    std::unique_ptr<DirectX::SpriteFont>    m_font;
    std::unique_ptr<DirectX::SpriteFont>    m_ctrlFont;
    Microsoft::WRL::ComPtr<ID3D11ShaderResourceView> m_background;

    //Gamepad states
    Platform::Collections::Vector<Windows::Gaming::Input::RawGameController^>^ m_localCollection;
	Platform::Collections::Vector<Windows::Gaming::Input::RacingWheel^>^ m_wheelCollection;

	Windows::Gaming::Input::ForceFeedback::ConstantForceEffect^ m_effect;
	Windows::Web::Http::HttpResponseMessage^ response;
	Windows::Web::Http::HttpClient^ httpClient;
    
    Windows::Gaming::Input::RawGameController^  m_currentController;
	Windows::Gaming::Input::RacingWheel^                m_currentWheel;
    uint32_t                                    m_currentButtonCount;
    uint32_t                                    m_currentSwitchCount;
    uint32_t                                    m_currentAxisCount;
    Platform::Array<bool>^             m_currentButtonReading;
	Platform::Array<Windows::Gaming::Input::GameControllerSwitchPosition>^ m_currentSwitchReading;
	Platform::Array<double>^           m_currentAxisReading;
	Windows::Gaming::Input::RacingWheelReading          m_wheelReading;

	
	bool                                                m_effectLoaded;
    std::wstring            m_buttonString;
    double                  m_leftTrigger;
    double                  m_rightTrigger;
    double                  m_leftStickX;
    double                  m_leftStickY;
    double                  m_rightStickX;
    double                  m_rightStickY;

    // Device resources.
    std::unique_ptr<DX::DeviceResources>    m_deviceResources;

    // Rendering loop timer.
    DX::StepTimer                           m_timer;
};