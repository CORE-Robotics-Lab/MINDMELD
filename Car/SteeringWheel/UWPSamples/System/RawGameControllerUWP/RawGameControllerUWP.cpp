//--------------------------------------------------------------------------------------
// RawGameControllerUWP.cpp
//
// Advanced Technology Group (ATG)
// Copyright (C) Microsoft Corporation. All rights reserved.
//--------------------------------------------------------------------------------------

#include "pch.h"
#include "RawGameControllerUWP.h"
#include <ppltasks.h>

#include "ATGColors.h"
#include "ControllerFont.h"



using namespace DirectX;
using namespace Windows::ApplicationModel;
using namespace Windows::Gaming::Input;
using namespace Windows::Foundation;
using namespace Windows::Foundation::Collections;
using namespace Platform::Collections;
using namespace Platform;
using namespace Windows::Storage;
using namespace Windows::Web::Http;
using namespace Windows::Web::Http::Headers;
using namespace Windows::Data::Json;


using Microsoft::WRL::ComPtr;


Sample::Sample()
{
    // Renders only 2D, so no need for a depth buffer.
    m_deviceResources = std::make_unique<DX::DeviceResources>(DXGI_FORMAT_B8G8R8A8_UNORM, DXGI_FORMAT_UNKNOWN);
    m_deviceResources->RegisterDeviceNotify(this);
}

// Initialize the Direct3D resources required to run.
void Sample::Initialize(IUnknown* window, int width, int height, DXGI_MODE_ROTATION rotation)
{
    m_deviceResources->SetWindow(window, width, height, rotation);

    m_deviceResources->CreateDeviceResources();  	
    CreateDeviceDependentResources();

    m_deviceResources->CreateWindowSizeDependentResources();
    CreateWindowSizeDependentResources();

	cal = ref new Windows::Globalization::Calendar();

	response = ref new HttpResponseMessage();
	httpClient = ref new HttpClient();

	//Create an effect
	m_effect = ref new ForceFeedback::ConstantForceEffect();
	TimeSpan time;
	time.Duration = 10000;
	Numerics::float3 vector;
	vector.x = 0.f;
	vector.y = 0.f;
	vector.z = 0.f;
	m_effect->SetParameters(vector, time);

    m_localCollection = ref new Vector<RawGameController^>();
	m_wheelCollection = ref new Vector<RacingWheel^>();

    auto gamecontrollers = RawGameController::RawGameControllers;
    for (auto gamecontroller : gamecontrollers)
    {
        m_localCollection->Append(gamecontroller);
		m_wheelCollection->Append(RacingWheel::FromGameController(gamecontroller));
    }

	RacingWheel::RacingWheelAdded += ref new EventHandler<RacingWheel^ >([=](Platform::Object^, RacingWheel^ args)
	{
		m_wheelCollection->Append(args);
		UpdateWheel();
	});

	RacingWheel::RacingWheelRemoved += ref new EventHandler<RacingWheel^ >([=](Platform::Object^, RacingWheel^ args)
	{
		unsigned int index;
		if (m_wheelCollection->IndexOf(args, &index))
		{
			m_wheelCollection->RemoveAt(index);
			UpdateWheel();
		}
	});

	RefreshControllerInfo();


    // UWP on Xbox One triggers a back request whenever the B button is pressed
    // which can result in the app being suspended if unhandled
    using namespace Windows::UI::Core;

    auto navigation = SystemNavigationManager::GetForCurrentView();

    navigation->BackRequested += ref new EventHandler<BackRequestedEventArgs^>([](Platform::Object^, BackRequestedEventArgs^ args)
    {
        args->Handled = true;
    });
}


#pragma region Frame Update
// Executes basic render loop.
void Sample::Tick()
{
    m_timer.Tick([&]()
    {
        Update(m_timer);
    });

    Render();
}

void Sample::UpdateController()
{
    RawGameController^ mostRecentController = nullptr;

    if (m_localCollection->Size > 0)
    {
        mostRecentController = m_localCollection->GetAt(0);
    }

	if (m_currentController != mostRecentController)
	{
		m_currentController = mostRecentController;
	}
}

void Sample::UpdateWheel()
{
	RacingWheel^ mostRecentWheel = nullptr;

	if (m_wheelCollection->Size > 0)
	{
		mostRecentWheel = m_wheelCollection->GetAt(0);
	}

	if (m_currentWheel != mostRecentWheel)
	{
		m_currentWheel = mostRecentWheel;
	}

	
	/*response = ref new HttpResponseMessage();
	httpClient = ref new HttpClient();
	Uri^ uri = ref new Uri("http://127.0.0.1:5000/ff/");
	Concurrency::create_task(httpClient->TryGetAsync(uri)).then([=](HttpRequestResult^ result)
	{
		if (result->Succeeded)
		{
			response = result->ResponseMessage;
			Concurrency::create_task(response->Content->ReadAsStringAsync()).then([=](String^ returnText)
			{
				String^ getStr = ref new String(returnText->Data());
				JsonObject^ getJSON = ref new JsonObject();
				getJSON = getJSON->Parse(getStr);
				TimeSpan time;
				time.Duration = 170000;
				//time.Duration = 10000000;
				Numerics::float3 vector;
				String^ name = "value";
				vector.x = getJSON->GetNamedNumber(name);
				vector.y = 0.f;
				vector.z = 0.f;
				m_effect->SetParameters(vector, time);
			});
		}
	});

	Uri^ uri_post = ref new Uri("http://127.0.0.1:5000/wheel/");
	HttpStringContent^ post_content = ref new HttpStringContent("test");
	Concurrency::create_task(httpClient->TryPostAsync(uri_post, post_content)).then([](HttpRequestResult^ result) {
		if (result->Succeeded)
		{
			String^ post = "win";
		}
	});*/

	if (m_currentWheel != nullptr && m_currentWheel->WheelMotor != nullptr)
	{

		m_effect = ref new ForceFeedback::ConstantForceEffect();

		IAsyncOperation<ForceFeedback::ForceFeedbackLoadEffectResult>^ request = m_currentWheel->WheelMotor->LoadEffectAsync(m_effect);

		auto loadEffectTask = Concurrency::create_task(request);
		loadEffectTask.then([=](ForceFeedback::ForceFeedbackLoadEffectResult result)
		{
			if (result == ForceFeedback::ForceFeedbackLoadEffectResult::Succeeded)
			{
				m_effectLoaded = true;
			}
			else
			{
				m_effectLoaded = false;
			}
		}).wait();

		if (m_effectLoaded)
		{
			m_effect->Start();
		}
	}

	RefreshControllerInfo();
}

void Sample::RefreshControllerInfo()
{
	if (m_currentController != nullptr)
	{
		m_currentButtonCount = m_currentController->ButtonCount;
		m_currentAxisCount = m_currentController->AxisCount;
		m_currentSwitchCount = m_currentController->SwitchCount;

		m_currentButtonReading = ref new Platform::Array<bool>(m_currentButtonCount);
		m_currentSwitchReading = ref new Platform::Array<GameControllerSwitchPosition>(m_currentSwitchCount);
		m_currentAxisReading = ref new Platform::Array<double>(m_currentAxisCount);
	}
}

// Updates the world.
void Sample::Update(DX::StepTimer const&)
{
	PIXBeginEvent(PIX_COLOR_DEFAULT, L"Update");

	bool toggleFFB = true;
	/*if (m_effect != nullptr) {
		if (m_effect->State == ForceFeedback::ForceFeedbackEffectState::Running)
		{
			toggleFFB = false;
		}
		else {
			toggleFFB = true;
		}
	}*/


	if (m_currentController == nullptr)
	{
		m_buttonString.clear();
		PIXEndEvent();
	}

	if (m_currentWheel != nullptr)
	{
		m_wheelReading = m_currentWheel->GetCurrentReading();
		JsonObject^ data = ref new JsonObject();

		JsonValue^ wheel_pos = JsonValue::CreateNumberValue(m_wheelReading.Wheel);
		JsonValue^ throttle = JsonValue::CreateNumberValue(m_wheelReading.Throttle);
		JsonValue^ brake = JsonValue::CreateNumberValue(m_wheelReading.Brake);
		//cal = ref new Windows::Globalization::Calendar();
		cal->SetToNow();

		double hr = cal->Hour;
		double min = cal->Minute;
		double s = cal->Second;
		double ns = cal->Nanosecond;
		double t = hr * 60 * 60 + min * 60 + s + ns / (1000000000);
		JsonValue^ t_data = JsonValue::CreateNumberValue(t);
		String^ wheel_label = "wheel";
		String^ throttle_label = "throttle";
		String^ brake_label = "brake";
		String^ time_label = "time";

		data->Insert(wheel_label, wheel_pos);
		data->Insert(throttle_label, throttle);
		data->Insert(brake_label, brake);
		data->Insert(time_label, t_data);

		Uri^ uri_post = ref new Uri("http://127.0.0.1:5000/data/");
		HttpStringContent^ str_data = ref new HttpStringContent(data->ToString());
		Concurrency::create_task(httpClient->TryPostAsync(uri_post, str_data)).then([](HttpRequestResult^ result) {
			if (result->Succeeded)
			{
				String^ post = "win";
			}
		});

		if (m_effectLoaded && toggleFFB)
		{
			Uri^ uri = ref new Uri("http://127.0.0.1:5000/ff/");
			Concurrency::create_task(httpClient->TryGetAsync(uri)).then([=](HttpRequestResult^ result)
			{
				if (result->Succeeded)
				{
					response = result->ResponseMessage;
					Concurrency::create_task(response->Content->ReadAsStringAsync()).then([=](String^ returnText)
					{
						String^ getStr = ref new String(returnText->Data());
						JsonObject^ getJSON = ref new JsonObject();
						getJSON = getJSON->Parse(getStr);
						TimeSpan time;
						//time.Duration = 10000000;
						time.Duration = 170000;
						Numerics::float3 vector;
						String^ name = "value";
						vector.x = static_cast<float>(getJSON->GetNamedNumber(name));
						vector.y = 0.f;
						vector.z = 0.f;
						m_effect->SetParameters(vector, time);
					});
				}
				else {
					TimeSpan time;
					//time.Duration = 10000000;
					time.Duration = 170000;
					Numerics::float3 vector;
					vector.x = 0.f;
					vector.y = 0.f;
					vector.z = 0.f;
					m_effect->SetParameters(vector, time);
				}
			});
			if (m_effect->State == ForceFeedback::ForceFeedbackEffectState::Running)
			{
				m_effect->Stop();
			}
			else 
			{
				m_effect->Start();
			}

			/*else
			{
				int rnd = rand();
				float ff = 0.f;
				if (rnd > RAND_MAX / 2)
				{
					ff = (float) -rnd / RAND_MAX;
				} 
				else
				{
					ff = (float) rnd / RAND_MAX;
				}

				TimeSpan time;
				time.Duration = 10000000;
				Numerics::float3 vector;
				vector.x = ff;
				vector.y = 0.f;
				vector.z = 0.f;
				m_effect->SetParameters(vector, time);
				m_effect->Start();
			}*/
		}
	}

		/*Uri^ uri_post_wheel = ref new Uri("http://127.0.0.1:5000/wheel/");
	HttpStringContent^ post_wheel = ref new HttpStringContent(m_wheelReading.Wheel.ToString());
	Concurrency::create_task(httpClient->TryPostAsync(uri_post_wheel, post_wheel)).then([](HttpRequestResult^ result) {
		if (result->Succeeded)
		{
			String^ post = "win";
		}
	});
	Uri^ uri_post_throttle = ref new Uri("http://127.0.0.1:5000/throttle/");
	HttpStringContent^ post_throttle = ref new HttpStringContent(m_wheelReading.Throttle.ToString());
	Concurrency::create_task(httpClient->TryPostAsync(uri_post_throttle, post_throttle)).then([](HttpRequestResult^ result) {
		if (result->Succeeded)
		{
			String^ post = "win";
		}
	});
	Uri^ uri_post_brake = ref new Uri("http://127.0.0.1:5000/brake/");
	HttpStringContent^ post_brake = ref new HttpStringContent(m_wheelReading.Brake.ToString());
	Concurrency::create_task(httpClient->TryPostAsync(uri_post_brake, post_brake)).then([](HttpRequestResult^ result) {
		if (result->Succeeded)
		{
			String^ post = "win";
		}
	});*/

    //GameControllerButtonLabel buttonLabel;

    //m_currentController->GetCurrentReading(
    //    m_currentButtonReading,
    //    m_currentSwitchReading,
   //     m_currentAxisReading);


    /*m_buttonString = L"Buttons pressed:  ";

    for (uint32_t i = 0; i < m_currentButtonCount; i++)
    {
        if (m_currentButtonReading[i])
        {
            buttonLabel = m_currentController->GetButtonLabel(i);

            switch (buttonLabel)
            {
            case GameControllerButtonLabel::XboxA:
                m_buttonString += L"[A] ";
                break;
            case GameControllerButtonLabel::XboxB:
                m_buttonString += L"[B] ";
                break;
            case GameControllerButtonLabel::XboxX:
                m_buttonString += L"[X] ";
                break;
            case GameControllerButtonLabel::XboxY:
                m_buttonString += L"[Y] ";
                break;
            case GameControllerButtonLabel::XboxLeftBumper:
                m_buttonString += L"[LB] ";
                break;
            case GameControllerButtonLabel::XboxRightBumper:
                m_buttonString += L"[RB] ";
                break;
            case GameControllerButtonLabel::XboxLeftStickButton:
                m_buttonString += L"[LThumb] ";
                break;
            case GameControllerButtonLabel::XboxRightStickButton:
                m_buttonString += L"[RThumb] ";
                break;
            case GameControllerButtonLabel::XboxMenu:
                m_buttonString += L"[Menu] ";
                break;
            case GameControllerButtonLabel::XboxView:
                m_buttonString += L"[View] ";
                break;
			case GameControllerButtonLabel::XboxUp:
				m_buttonString += L"[DPad]Up ";
				break;
			case GameControllerButtonLabel::XboxDown:
				m_buttonString += L"[DPad]Down ";
				break;
			case GameControllerButtonLabel::XboxLeft:
				m_buttonString += L"[DPad]Left ";
				break;
			case GameControllerButtonLabel::XboxRight:
				m_buttonString += L"[DPad]Right ";
				break;
			}
        }
    }

    for (uint32_t i = 0; i < m_currentSwitchCount; i++)
    {
		//Handle m_currentSwitchReading[i], reading GameControllerSwitchPosition
    }

	if (m_currentAxisCount == 6)
	{
		//Xbox controllers have 6 axis: 2 for each stick and one for each trigger
		m_leftStickX = m_currentAxisReading[0];
		m_leftStickY = m_currentAxisReading[1];
		m_rightStickX = m_currentAxisReading[2];
		m_rightStickY = m_currentAxisReading[3];
		m_leftTrigger = m_currentAxisReading[4];
		m_rightTrigger = m_currentAxisReading[5];
	}*/

    PIXEndEvent();
}
#pragma endregion

#pragma region Frame Render
// Draws the scene.
void Sample::Render()
{
    // Don't try to render anything before the first Update.
    if (m_timer.GetFrameCount() == 0)
    {
        return;
    }

    Clear();

    auto context = m_deviceResources->GetD3DDeviceContext();
    PIXBeginEvent(context, PIX_COLOR_DEFAULT, L"Render");

    auto rect = m_deviceResources->GetOutputSize();
    auto safeRect = SimpleMath::Viewport::ComputeTitleSafeArea(rect.right, rect.bottom);

    XMFLOAT2 pos(float(safeRect.left), float(safeRect.top));
    //wchar_t tempString[256] = {};

    m_spriteBatch->Begin();

    m_spriteBatch->Draw(m_background.Get(), m_deviceResources->GetOutputSize());
    
    /*if (!m_buttonString.empty())
    {
        DX::DrawControllerString(m_spriteBatch.get(), m_font.get(), m_ctrlFont.get(), m_buttonString.c_str(), pos);
        pos.y += m_font->GetLineSpacing() * 1.5f;

        swprintf(tempString, 255, L"[LT]  %1.3f", m_leftTrigger);
        DX::DrawControllerString(m_spriteBatch.get(), m_font.get(), m_ctrlFont.get(), tempString, pos);
        pos.y += m_font->GetLineSpacing() * 1.5f;

        swprintf(tempString, 255, L"[RT]  %1.3f", m_rightTrigger);
        DX::DrawControllerString(m_spriteBatch.get(), m_font.get(), m_ctrlFont.get(), tempString, pos);
        pos.y += m_font->GetLineSpacing() * 1.5f;

        swprintf(tempString, 255, L"[LThumb]  X: %1.3f  Y: %1.3f", m_leftStickX, m_leftStickY);
        DX::DrawControllerString(m_spriteBatch.get(), m_font.get(), m_ctrlFont.get(), tempString, pos);
        pos.y += m_font->GetLineSpacing() * 1.5f;

        swprintf(tempString, 255, L"[RThumb]  X: %1.3f  Y: %1.3f", m_rightStickX, m_rightStickY);
        DX::DrawControllerString(m_spriteBatch.get(), m_font.get(), m_ctrlFont.get(), tempString, pos);
    }
    else
    {
        m_font->DrawString(m_spriteBatch.get(), L"No controller connected", pos, ATG::Colors::Orange);
    }*/


	if (m_currentWheel != nullptr)
	{
		DrawWheel(pos);
	}
	else
	{
		m_font->DrawString(m_spriteBatch.get(), L"No wheel connected", pos, ATG::Colors::Orange);
	}

    m_spriteBatch->End();

    PIXEndEvent(context);

    // Show the new frame.
    PIXBeginEvent(PIX_COLOR_DEFAULT, L"Present");
    m_deviceResources->Present();
    PIXEndEvent();
}

void Sample::DrawWheel(XMFLOAT2 startPosition)
{
	wchar_t wheelString[128] = {};

	swprintf_s(wheelString, L"Wheel %1.3f", m_wheelReading.Wheel);
	m_font->DrawString(m_spriteBatch.get(), wheelString, startPosition, ATG::Colors::Green);
	startPosition.y += m_font->GetLineSpacing() * 1.1f;

	swprintf_s(wheelString, L"Throttle %1.3f", m_wheelReading.Throttle);
	m_font->DrawString(m_spriteBatch.get(), wheelString, startPosition, ATG::Colors::Green);
	startPosition.y += m_font->GetLineSpacing() * 1.1f;

	swprintf_s(wheelString, L"Break %1.3f", m_wheelReading.Brake);
	m_font->DrawString(m_spriteBatch.get(), wheelString, startPosition, ATG::Colors::Green);
	startPosition.y += m_font->GetLineSpacing() * 1.1f;

	if (m_currentWheel->HasClutch)
	{
		swprintf_s(wheelString, L"Clutch %1.3f", m_wheelReading.Clutch);
		m_font->DrawString(m_spriteBatch.get(), wheelString, startPosition, ATG::Colors::Green);
		startPosition.y += m_font->GetLineSpacing() * 1.1f;
	}

	if (m_currentWheel->HasHandbrake)
	{
		swprintf_s(wheelString, L"Handbrake %1.3f", m_wheelReading.Handbrake);
		m_font->DrawString(m_spriteBatch.get(), wheelString, startPosition, ATG::Colors::Green);
		startPosition.y += m_font->GetLineSpacing() * 1.1f;
	}

	if (m_currentWheel->HasPatternShifter)
	{
		swprintf_s(wheelString, L"Shifter %d of %d", m_wheelReading.PatternShifterGear, m_currentWheel->MaxPatternShifterGear);
		m_font->DrawString(m_spriteBatch.get(), wheelString, startPosition, ATG::Colors::Green);
		startPosition.y += m_font->GetLineSpacing() * 1.1f;
	}
}

// Helper method to clear the back buffers.
void Sample::Clear()
{
    auto context = m_deviceResources->GetD3DDeviceContext();
    PIXBeginEvent(context, PIX_COLOR_DEFAULT, L"Clear");

    // Clear the views.
    auto renderTarget = m_deviceResources->GetRenderTargetView();

    context->ClearRenderTargetView(renderTarget, ATG::Colors::Background);

    context->OMSetRenderTargets(1, &renderTarget, nullptr);

    // Set the viewport.
    auto viewport = m_deviceResources->GetScreenViewport();
    context->RSSetViewports(1, &viewport);

    PIXEndEvent(context);
}
#pragma endregion

#pragma region Message Handlers
// Message handlers
void Sample::OnActivated()
{
}

void Sample::OnDeactivated()
{
}

void Sample::OnSuspending()
{
    auto context = m_deviceResources->GetD3DDeviceContext();
    context->ClearState();

    m_deviceResources->Trim();
}

void Sample::OnResuming()
{
    m_timer.ResetElapsedTime();
}

void Sample::OnWindowSizeChanged(int width, int height, DXGI_MODE_ROTATION rotation)
{
    if (!m_deviceResources->WindowSizeChanged(width, height, rotation))
        return;

    CreateWindowSizeDependentResources();
}

void Sample::ValidateDevice()
{
    m_deviceResources->ValidateDevice();
}

// Properties
void Sample::GetDefaultSize(int& width, int& height) const
{
    width = 1280;
    height = 720;
}
#pragma endregion

#pragma region Direct3D Resources
// These are the resources that depend on the device.
void Sample::CreateDeviceDependentResources()
{
    auto context = m_deviceResources->GetD3DDeviceContext();
    auto device = m_deviceResources->GetD3DDevice();

    m_spriteBatch = std::make_unique<SpriteBatch>(context);

    m_font = std::make_unique<SpriteFont>(device, L"SegoeUI_24.spritefont");
    m_ctrlFont = std::make_unique<SpriteFont>(device, L"XboxOneController.spritefont");

    DX::ThrowIfFailed(CreateDDSTextureFromFile(device, L"gamepad.dds", nullptr, m_background.ReleaseAndGetAddressOf()));
}

// Allocate all memory resources that change on a window SizeChanged event.
void Sample::CreateWindowSizeDependentResources()
{
    m_spriteBatch->SetRotation(m_deviceResources->GetRotation());
}

void Sample::OnDeviceLost()
{
    m_spriteBatch.reset();
    m_font.reset();
    m_ctrlFont.reset();
    m_background.Reset();
}

void Sample::OnDeviceRestored()
{
    CreateDeviceDependentResources();

    CreateWindowSizeDependentResources();
}
#pragma endregion
